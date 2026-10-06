"""
Module to support training a model w.r.t. a dataset.
"""

import itertools
import logging
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from profiler import profiler, torch_profile
from torch import nn

from semi_sup_seg.checkpoints import get_checkpoint_file_name, save_checkpoint
from semi_sup_seg.constants import MAX_PIXEL_INT_VALUE
from semi_sup_seg.data.augmentation import Augmentation, AugmentationSampler
from semi_sup_seg.data.datapoints import LoadedDatapoint, batch_datapoints
from semi_sup_seg.data.iteration import TensorDatapointIter, prefetch
from semi_sup_seg.data.labels import LabelMetadata
from semi_sup_seg.inference import PseudoLabeler
from semi_sup_seg.metrics import (
    compute_public_benchmark_metrics,
    get_batch_confusion_matrix,
    get_empty_confusion_matrix,
)
from semi_sup_seg.precision import DEFAULT_MIXED_PRECISION, autocast
from semi_sup_seg.train_logging import (
    DEFAULT_LOG_EVERY_N,
    DEFAULT_LOG_IMAGE_COUNT,
    TrainingLogger,
)

logger = logging.getLogger(__name__)


def _get_synthetic_batch(
    batch_size: int, height: int, width: int, label_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get uint8 (B, H, W, 3) all-black image and (B, H, W) all-`label_id` label.
    """

    image_batch = torch.zeros((batch_size, height, width, 3), dtype=torch.uint8)
    label_batch = torch.full(
        (batch_size, height, width), label_id, dtype=torch.uint8
    )
    return image_batch, label_batch


class Trainer:
    """
    Class to support training a model w.r.t. a dataset.
    """

    def __init__(
        self,
        *,
        labeled_image_size: tuple[int, int],
        labeled_train_data_iter: Iterator[LoadedDatapoint],
        labeled_val_data_iter: Iterator[LoadedDatapoint],
        unlabeled_train_data_iter: Iterator[LoadedDatapoint] | None = None,
        unlabeled_image_size: tuple[int, int] | None = None,
        label_metadata: LabelMetadata,
        pseudo_labeler: PseudoLabeler | None = None,
        augmentation_sampler: AugmentationSampler | None = None,
        train_batch_size: int,
        val_batch_size: int,
        loss_fn: Callable[..., torch.Tensor],
        mixed_precision: bool = DEFAULT_MIXED_PRECISION,
        log_dir: Path | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ):
        # Data, as read from filesystem.
        self.labeled_image_size = labeled_image_size
        self.labeled_train_data_iter = labeled_train_data_iter
        self.labeled_val_data_iter = labeled_val_data_iter
        self.unlabeled_train_data_iter = unlabeled_train_data_iter
        self.unlabeled_image_size = unlabeled_image_size
        self.label_metadata = label_metadata

        # Data transformation.
        self.pseudo_labeler = pseudo_labeler
        self.augmentation_sampler = augmentation_sampler

        # Other.
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.loss_fn = loss_fn
        self.mixed_precision = mixed_precision
        self.log_dir = log_dir
        self.checkpoint_metadata = checkpoint_metadata or {}

        if pseudo_labeler is not None and (
            unlabeled_train_data_iter is None or unlabeled_image_size is None
        ):
            raise ValueError("Pseudo-labeling needs unlabeled train data.")

        self.label_map = label_metadata.get_raw_to_train_label_map()

        # Scales loss before backward, so fp16 grads don't underflow, then
        # unscales before optimizer step.
        self.grad_scaler = torch.amp.GradScaler(
            "cuda", enabled=mixed_precision and torch.cuda.is_available()
        )

        self.training_logger = TrainingLogger(
            log_dir=log_dir, label_metadata=label_metadata
        )

    def _get_mixed_batch_sizes(self) -> tuple[int, int]:
        """
        Get the human-labeled & unlabeled counts in a pseudo-labeling batch.
        """

        labeled_part, unlabeled_part = (
            self.pseudo_labeler.config.labeled_to_unlabeled_ratio
        )
        labeled_batch_size = (
            self.train_batch_size
            * labeled_part
            // (labeled_part + unlabeled_part)
        )
        return labeled_batch_size, self.train_batch_size - labeled_batch_size

    def _get_datapoint_loss_weights(
        self, labeled_count: int, unlabeled_count: int, device: torch.device
    ) -> torch.Tensor | None:
        """
        Get (B,) loss multipliers for a batch: labeled, then unlabeled.
        """

        if unlabeled_count == 0:
            return None

        labeled_multiplier, unlabeled_multiplier = (
            self.pseudo_labeler.config.labeled_and_unlabeled_loss_multipliers
        )
        return torch.cat(
            [
                torch.full((labeled_count,), labeled_multiplier, device=device),
                torch.full(
                    (unlabeled_count,), unlabeled_multiplier, device=device
                ),
            ]
        )

    def _iterate_data(
        self,
        data_iter: Iterator[LoadedDatapoint],
        datapoint_count: int,
        batch_size: int,
        profile_phase: str = "load raw data",
    ) -> Iterator[tuple[torch.Tensor, ...]]:
        """
        Yield batches from `data_iter` until `datapoint_count` datapoints yielded.
        """

        if datapoint_count <= 0:
            return

        remaining = datapoint_count
        iterator = prefetch(
            batch_datapoints(
                itertools.islice(data_iter, datapoint_count), batch_size
            )
        )

        while True:
            with profiler.phase(profile_phase):
                next_batch = next(iterator, None)

            if next_batch is None:
                break

            remaining -= len(next_batch[0])
            yield next_batch
            if remaining == 0:
                return

        raise ValueError(
            f"Data ran out after {(datapoint_count - remaining):,} datapoints. "
            f"Requested {datapoint_count:,}."
        )

    def _iterate_train_data(
        self, datapoint_count: int
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
        """
        Yield (image, label, unlabeled image) train batches.
        """

        warmup_datapoint_count = datapoint_count
        if self.pseudo_labeler is not None:
            config = self.pseudo_labeler.config
            warmup_datapoint_count = min(
                datapoint_count, config.warmup_datapoints_before_pseudo_labeling
            )

        for image_batch, label_batch in self._iterate_data(
            self.labeled_train_data_iter,
            warmup_datapoint_count,
            self.train_batch_size,
        ):
            yield image_batch, label_batch, None

        mixed_batch_count = (
            datapoint_count - warmup_datapoint_count
        ) // self.train_batch_size
        if mixed_batch_count == 0:
            return

        labeled_batch_size, unlabeled_batch_size = self._get_mixed_batch_sizes()
        labeled_batches = self._iterate_data(
            self.labeled_train_data_iter,
            mixed_batch_count * labeled_batch_size,
            labeled_batch_size,
        )
        unlabeled_batches = self._iterate_data(
            self.unlabeled_train_data_iter,
            mixed_batch_count * unlabeled_batch_size,
            unlabeled_batch_size,
            profile_phase="load raw unlabeled data",
        )
        for (image_batch, label_batch), (unlabeled_image_batch,) in zip(
            labeled_batches, unlabeled_batches, strict=True
        ):
            yield image_batch, label_batch, unlabeled_image_batch

    def _iterate_val_data(self, datapoint_count: int) -> TensorDatapointIter:
        return self._iterate_data(
            self.labeled_val_data_iter, datapoint_count, self.val_batch_size
        )

    @staticmethod
    def _to_images(
        image_batch: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Convert a uint8 image batch to model-ready images on device.
        """

        # uint8 (B, H, W, 3) --> float (B, 3, H, W) in [0, 1].
        images = image_batch.to(device, non_blocking=True)
        return images.permute(0, 3, 1, 2).float() / MAX_PIXEL_INT_VALUE

    def _to_tensors(
        self,
        image_batch: torch.Tensor,
        label_batch: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert uint8 batches to model-ready tensors on device.
        """

        # uint8 (B, H, W) --> int64 (B, H, W) training labels.
        labels = label_batch.to(device, non_blocking=True)

        if self.label_map.device != device:
            self.label_map = self.label_map.to(device)
        labels = self.label_map[labels.int()]

        images = self._to_images(image_batch, device)

        return images, labels

    def _to_train_tensors(
        self,
        image_batch: torch.Tensor,
        label_batch: torch.Tensor,
        device: torch.device,
        unlabeled_image_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Augmentation | None]:
        """
        Like `_to_tensors`, but potentially augmented & auto-labeled.
        """

        images, labels = self._to_tensors(image_batch, label_batch, device)
        if unlabeled_image_batch is not None:
            unlabeled_images = self._to_images(unlabeled_image_batch, device)
            images = torch.cat([images, unlabeled_images])
            labels = torch.cat(
                [labels, self.pseudo_labeler.label(unlabeled_images)]
            )

        if self.augmentation_sampler is None:
            return images, labels, None

        return self.augmentation_sampler.augment(
            images, labels, ignore_id=self.label_metadata.ignore_id
        )

    def _loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        datapoint_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.loss_fn(
            logits, labels, self.label_metadata.ignore_id, datapoint_weights
        )

    @torch.no_grad()
    def _val_step(
        self,
        model: nn.Module,
        image_batch: torch.Tensor,
        label_batch: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate on a batch, do not update the model.
        """

        images, labels = self._to_tensors(image_batch, label_batch, device)
        with autocast(device, self.mixed_precision):
            logits = model(images)
            datapoint_losses = torch.stack(
                [
                    self._loss(logits[i : i + 1], labels[i : i + 1])
                    for i in range(len(images))
                ]
            )

        return images, labels, logits, datapoint_losses

    def _warm_up_step(
        self,
        model: nn.Module,
        image_batch: torch.Tensor,
        label_batch: torch.Tensor,
        device: torch.device,
        unlabeled_image_batch: torch.Tensor | None = None,
    ) -> None:
        """
        Evaluate on a batch, do the backward pass, but do not update model.
        """

        images, labels, _ = self._to_train_tensors(
            image_batch, label_batch, device, unlabeled_image_batch
        )
        datapoint_weights = self._get_datapoint_loss_weights(
            len(image_batch), len(images) - len(image_batch), device
        )
        with autocast(device, self.mixed_precision):
            loss = self._loss(model(images), labels, datapoint_weights)

        loss.backward()
        model.zero_grad(set_to_none=True)

    def _train_step(
        self,
        model: nn.Module,
        image_batch: torch.Tensor,
        label_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        unlabeled_image_batch: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Augmentation | None,
    ]:
        """
        Evaluate on a batch, and update the model.
        """

        images, labels, augmentation = self._to_train_tensors(
            image_batch, label_batch, device, unlabeled_image_batch
        )
        datapoint_weights = self._get_datapoint_loss_weights(
            len(image_batch), len(images) - len(image_batch), device
        )
        with autocast(device, self.mixed_precision):
            logits = model(images)
            loss = self._loss(logits, labels, datapoint_weights)

        optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(optimizer)
        self.grad_scaler.update()

        return images, labels, logits.detach(), loss.detach(), augmentation

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        datapoint_count: int,
        log_train_datapoints_seen: int,
        log_train_total_datapoint_count: int,
        log_image_count: int = DEFAULT_LOG_IMAGE_COUNT,
    ) -> dict[str, float] | None:
        """
        Validate `model` w.r.t. `datapoint_count` sampled val datapoints.
        """

        device = next(model.parameters()).device
        was_training = model.training

        datapoints_seen = 0
        datapoint_losses: list[torch.Tensor] = []
        confusion = get_empty_confusion_matrix(self.label_metadata, device)

        try:
            model.eval()

            for image_batch, label_batch in self._iterate_val_data(
                datapoint_count
            ):
                is_logging_step = datapoints_seen == 0

                with profiler.phase("infer on val batch", on_gpu=True):
                    images, labels, logits, batch_losses = self._val_step(
                        model, image_batch, label_batch, device
                    )

                    datapoints_seen += len(images)
                    datapoint_losses.append(batch_losses)
                    confusion += get_batch_confusion_matrix(
                        labels, logits.argmax(dim=1), self.label_metadata
                    )

                if not is_logging_step:
                    continue

                with profiler.phase("log for val batch"):
                    self.training_logger.log_images(
                        log_base_name="val",
                        datapoints_seen=log_train_datapoints_seen,
                        images=images,
                        labels=labels,
                        logits=logits,
                        num_images_to_log=log_image_count,
                    )
        finally:
            model.train(was_training)

        if datapoints_seen == 0:
            return None

        with profiler.phase("log for val batch"):
            metrics = compute_public_benchmark_metrics(
                confusion, self.label_metadata
            )
            all_losses = torch.cat(datapoint_losses).float()
            self.training_logger.log_val_metrics(
                datapoints_seen=log_train_datapoints_seen,
                total_datapoint_count=log_train_total_datapoint_count,
                loss_avg=all_losses.mean().item(),
                loss_stddev=all_losses.std(correction=0).item(),
                metrics=metrics,
            )

        return metrics

    def warm_up(self, model: nn.Module) -> None:
        """
        Prepare a model for training.
        """

        device = next(model.parameters()).device
        was_training = model.training

        is_eval = self.label_map != self.label_metadata.ignore_id
        label_id = int(is_eval.nonzero()[0])
        train_batch = _get_synthetic_batch(
            self.train_batch_size, *self.labeled_image_size, label_id
        )
        val_batch = _get_synthetic_batch(
            self.val_batch_size, *self.labeled_image_size, label_id
        )

        try:
            model.train()
            self._warm_up_step(model, *train_batch, device)

            if self.pseudo_labeler is not None:
                labeled_batch_size, unlabeled_batch_size = (
                    self._get_mixed_batch_sizes()
                )
                self._warm_up_step(
                    model,
                    *_get_synthetic_batch(
                        labeled_batch_size, *self.labeled_image_size, label_id
                    ),
                    device,
                    unlabeled_image_batch=_get_synthetic_batch(
                        unlabeled_batch_size,
                        *self.unlabeled_image_size,
                        label_id,
                    )[0],
                )

            model.eval()
            self._val_step(model, *val_batch, device)
        finally:
            model.train(was_training)

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        datapoints_seen: int,
        total_datapoint_count: int,
        val_metrics: dict[str, float] | None,
    ) -> None:
        """
        Save model & optimizer state under `self.log_dir`, if it is set.
        """

        if self.log_dir is None:
            return

        path = self.log_dir / get_checkpoint_file_name(
            datapoints_seen, total_datapoint_count
        )
        with profiler.phase("save checkpoint"):
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                grad_scaler=self.grad_scaler,
                metadata={
                    **self.checkpoint_metadata,
                    "datapoints_seen": datapoints_seen,
                    "val_metrics": val_metrics,
                    "teacher_datapoints_seen": (
                        self.pseudo_labeler.teacher_datapoints_seen
                        if self.pseudo_labeler is not None
                        else None
                    ),
                },
                teacher=self._get_teacher(),
            )

        logger.info(f"Saved checkpoint to '{path}'.")

    def _get_teacher(self) -> nn.Module | None:
        """
        Get the pseudo-labeler, if any.
        """

        if (
            self.pseudo_labeler is None
            or self.pseudo_labeler.teacher_datapoints_seen is None
        ):
            return None
        return self.pseudo_labeler.model

    def _update_teacher(self, model: nn.Module, datapoints_seen: int) -> None:
        """
        Let the pseudo-labeler, if any, update its teacher from `model`.
        """

        if self.pseudo_labeler is not None:
            self.pseudo_labeler.update_teacher(model, datapoints_seen)

    def train(
        self,
        model: nn.Module,
        train_datapoint_count: int,
        val_datapoint_count: int,
        optimizer: torch.optim.Optimizer,
        log_every_n_datapoints: int = DEFAULT_LOG_EVERY_N,
        log_image_count: int = DEFAULT_LOG_IMAGE_COUNT,
    ) -> None:
        """
        Train `model` w.r.t. `train_datapoint_count` sampled train datapoints.
        """

        device = next(model.parameters()).device
        model.train()

        datapoints_seen = 0
        loss_sum_since_last_log = torch.zeros((), device=device)
        augmentations_since_last_log: list[Augmentation] = []

        datapoints_seen_at_last_log = 0
        datapoint_index_for_next_log = log_every_n_datapoints
        datapoints_seen_at_last_checkpoint = None

        try:
            val_metrics = self.validate(
                model=model,
                datapoint_count=val_datapoint_count,
                log_train_datapoints_seen=0,
                log_train_total_datapoint_count=train_datapoint_count,
                log_image_count=log_image_count,
            )
            self._save_checkpoint(
                model,
                optimizer,
                datapoints_seen=0,
                total_datapoint_count=train_datapoint_count,
                val_metrics=val_metrics,
            )
            datapoints_seen_at_last_checkpoint = 0
            self._update_teacher(model, datapoints_seen=0)
            time_of_last_log = time.perf_counter()

            with torch_profile() as end_torch_profile_step:
                for (
                    image_batch,
                    label_batch,
                    unlabeled_image_batch,
                ) in self._iterate_train_data(train_datapoint_count):
                    with profiler.phase("infer on train batch", on_gpu=True):
                        images, labels, logits, loss, augmentation = (
                            self._train_step(
                                model,
                                image_batch,
                                label_batch,
                                optimizer,
                                device,
                                unlabeled_image_batch,
                            )
                        )

                        datapoints_seen += len(images)
                        loss_sum_since_last_log += loss * len(images)
                        if augmentation is not None:
                            augmentations_since_last_log.append(augmentation)

                    end_torch_profile_step()

                    if not (
                        datapoints_seen >= datapoint_index_for_next_log
                        or datapoints_seen == train_datapoint_count
                    ):
                        continue

                    datapoints_since_last_log = (
                        datapoints_seen - datapoints_seen_at_last_log
                    )
                    mean_loss_since_last_log = (
                        loss_sum_since_last_log / datapoints_since_last_log
                    ).item()
                    time_since_last_log = time.perf_counter() - time_of_last_log

                    with profiler.phase("log for train batch"):
                        self.training_logger.log_train_metrics(
                            datapoints_seen=datapoints_seen,
                            total_datapoint_count=train_datapoint_count,
                            datapoints_per_second=(
                                datapoints_since_last_log / time_since_last_log
                            ),
                            loss=mean_loss_since_last_log,
                            learning_rate=optimizer.param_groups[0]["lr"],
                        )
                        if augmentations_since_last_log:
                            self.training_logger.log_augmentations(
                                datapoints_seen=datapoints_seen,
                                augmentation=Augmentation.cat(
                                    augmentations_since_last_log
                                ),
                            )
                        self.training_logger.log_images(
                            log_base_name="train",
                            datapoints_seen=datapoints_seen,
                            images=images,
                            labels=labels,
                            logits=logits,
                            num_images_to_log=log_image_count,
                        )

                    val_metrics = self.validate(
                        model=model,
                        datapoint_count=val_datapoint_count,
                        log_train_datapoints_seen=datapoints_seen,
                        log_train_total_datapoint_count=train_datapoint_count,
                        log_image_count=log_image_count,
                    )
                    self._save_checkpoint(
                        model,
                        optimizer,
                        datapoints_seen=datapoints_seen,
                        total_datapoint_count=train_datapoint_count,
                        val_metrics=val_metrics,
                    )
                    datapoints_seen_at_last_checkpoint = datapoints_seen
                    self._update_teacher(model, datapoints_seen)

                    loss_sum_since_last_log.zero_()
                    augmentations_since_last_log.clear()

                    time_of_last_log = time.perf_counter()
                    datapoints_seen_at_last_log = datapoints_seen
                    datapoint_index_for_next_log = (
                        datapoints_seen // log_every_n_datapoints + 1
                    ) * log_every_n_datapoints
        finally:
            self.training_logger.close()

            if datapoints_seen != datapoints_seen_at_last_checkpoint:
                self._save_checkpoint(
                    model,
                    optimizer,
                    datapoints_seen=datapoints_seen,
                    total_datapoint_count=train_datapoint_count,
                    val_metrics=None,
                )
