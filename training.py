"""
Module to support training segmentation models on datasets.

Example usage:
PROFILE_RESOURCES=1 PROFILE_RESOURCES_TORCH=1 python training.py $LABEL
"""

import dataclasses
import itertools
import logging
import math
import os
import sys
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from profiler import profiler, torch_profile
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from checkpoints import (
    get_checkpoint_file_name,
    get_init_checkpoint_metadata,
    load_unet_checkpoint,
    save_checkpoint,
)
from constants import (
    CITYSCAPES_IMAGE_HEIGHT,
    CITYSCAPES_IMAGE_WIDTH,
    DEFAULT_RUNS_DIR,
    DEFAULT_SEED,
    MAX_LABEL_COUNT,
    MAX_PIXEL_INT_VALUE,
)
from data import (
    CityscapesDatapoint,
    CityscapesLabeledDataset,
    CityscapesPersonLabeledDataset,
    LabelMetadata,
)
from data_iter import TensorDatapointIter, batch, prefetch
from metrics import (
    compute_public_benchmark_metrics,
    get_batch_confusion_matrix,
    get_empty_confusion_matrix,
)
from models import UNet
from precision import DEFAULT_MIXED_PRECISION, autocast

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_DOWNSCALE = 2

DEFAULT_BASE_HEIGHT = 256
DEFAULT_BASE_WIDTH = 512

DEFAULT_LOG_EVERY_N = 500
DEFAULT_LOG_IMAGE_COUNT = 2
DEFAULT_LOG_IMAGE_DOWNSCALE = 4


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


class TrainingLogger:
    """
    Class to support logging training state.
    """

    def __init__(self, log_dir: Path | None, label_metadata: LabelMetadata):
        self.writer = SummaryWriter(log_dir) if log_dir is not None else None

        # Label --> RGB in [0, 1].
        self.palette = torch.zeros((MAX_LABEL_COUNT, 3))
        for class_id, color in label_metadata.class_colors.items():
            self.palette[class_id] = torch.tensor(color) / MAX_PIXEL_INT_VALUE

    @staticmethod
    def _shrink(
        x: torch.Tensor, height: int, width: int, mode: str
    ) -> torch.Tensor:
        """
        Resize a (N, C, H, W) tensor to (N, C, height, width).
        """

        if mode == "nearest":
            return F.interpolate(x, size=(height, width), mode="nearest")
        return F.interpolate(x, size=(height, width), mode=mode, antialias=True)

    def _classes_to_rgb(
        self, labels: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        Convert (N, H, W) labels to a (N, 3, height, width) color image.
        """

        small_labels = self._shrink(
            labels[:, None].float(), height, width, mode="nearest"
        )[:, 0].long()

        palette = self.palette.to(labels.device)
        return palette[small_labels].permute(0, 3, 1, 2)

    @staticmethod
    def _confidences_to_rgb(
        logits: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        Convert (N, C, H, W) logits to a (N, 3, height, width) grayscale image.
        """

        small_logits = TrainingLogger._shrink(
            logits, height, width, mode="bilinear"
        )
        confidences = small_logits.softmax(dim=1).amax(dim=1, keepdim=True)

        return confidences.expand(-1, 3, -1, -1)

    def log_train_metrics(
        self,
        datapoints_seen: int,
        total_datapoint_count: int,
        datapoints_per_second: float,
        loss: float,
        learning_rate: float,
    ) -> None:
        """
        Log core training metrics.
        """

        # fmt: off
        logger.info(
            "\n".join(
                [
                    "train logs:",
                    f"\tdatapoints seen: {datapoints_seen:,}/{total_datapoint_count:,}",
                    f"\tdatapoints per second: {datapoints_per_second:.1f}",
                    f"\tloss: {loss:.4f}",
                    f"\tlearning rate: {learning_rate:.2e}",
                ]
            )
        )
        # fmt: on

        if self.writer is not None:
            for name, value in [
                ("train/datapoints_per_second", datapoints_per_second),
                ("train/loss", loss),
                ("train/learning_rate", learning_rate),
            ]:
                self.writer.add_scalar(name, value, datapoints_seen)

    def log_val_metrics(
        self,
        datapoints_seen: int,
        total_datapoint_count: int,
        loss: float,
        metrics: dict[str, float],
    ) -> None:
        """
        Log core validation metrics.

        `metrics` comes from `metrics.compute_public_benchmark_metrics`.
        """

        # fmt: off
        logger.info(
            "\n".join(
                [
                    "val logs:",
                    f"\t(train) datapoints seen: {datapoints_seen:,}/{total_datapoint_count:,}",
                    f"\tloss: {loss:.4f}",
                    f"\taccuracy: {metrics['accuracy']:.2%}",
                    f"\tmean IoU: {metrics['mean_iou']:.2%}",
                ]
            )
        )
        # fmt: on

        if self.writer is not None:
            self.writer.add_scalar("val/loss", loss, datapoints_seen)
            for name, value in metrics.items():
                if not math.isnan(value):
                    self.writer.add_scalar(
                        f"val/{name}", value, datapoints_seen
                    )

    @torch.no_grad()
    def log_images(
        self,
        log_base_name: str,
        datapoints_seen: int,
        images: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        num_images_to_log: int,
        image_downscale: int = DEFAULT_LOG_IMAGE_DOWNSCALE,
    ) -> None:
        """
        Log images, true & predicted classes, and prediction confidences.
        """

        if self.writer is None or num_images_to_log <= 0:
            return

        images = images[:num_images_to_log]
        labels = labels[:num_images_to_log]
        logits = logits[:num_images_to_log].float()

        predictions = logits.argmax(dim=1)

        height = images.shape[-2] // image_downscale
        width = images.shape[-1] // image_downscale

        # Each element of `panels` has shape (N, 3, height, width).
        panels = [
            self._shrink(images, height, width, mode="bilinear").clamp(0, 1),
            self._classes_to_rgb(labels, height, width),
            self._classes_to_rgb(predictions, height, width),
            self._confidences_to_rgb(logits, height, width),
        ]
        # `rows` has shape (N, 3, height, panel_count * width).
        rows = torch.cat(panels, dim=-1)
        # `grid` has shape (3, N * height, panel_count * width).
        grid = torch.cat(list(rows), dim=-2)
        self.writer.add_image(
            f"{log_base_name}/samples", grid.cpu(), datapoints_seen
        )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


class Trainer:
    """
    Class to support training a model w.r.t. a dataset.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        train_data_iter: Iterator[CityscapesDatapoint],
        val_data_iter: Iterator[CityscapesDatapoint],
        train_batch_size: int,
        val_batch_size: int,
        label_metadata: LabelMetadata,
        log_dir: Path | None,
        train_downscale: int = DEFAULT_TRAIN_DOWNSCALE,
        mixed_precision: bool = DEFAULT_MIXED_PRECISION,
        checkpoint_metadata: dict[str, Any] | None = None,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.train_data_iter = train_data_iter
        self.val_data_iter = val_data_iter
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.label_metadata = label_metadata
        self.log_dir = log_dir
        self.train_downscale = train_downscale
        self.mixed_precision = mixed_precision
        self.checkpoint_metadata = checkpoint_metadata or {}

        # Raw label --> training label.
        self.label_map = torch.full(
            (MAX_LABEL_COUNT,), label_metadata.ignore_id, dtype=torch.long
        )
        for class_id in label_metadata.eval_class_ids:
            self.label_map[class_id] = class_id

        # Scales loss before backward, so fp16 grads don't underflow, then
        # unscales before optimizer step.
        self.grad_scaler = torch.amp.GradScaler(
            "cuda", enabled=mixed_precision and torch.cuda.is_available()
        )

        self.training_logger = TrainingLogger(
            log_dir=log_dir, label_metadata=label_metadata
        )

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

        The file name holds `datapoints_seen`, the train datapoints seen so far.
        The metadata holds it too, along with `val_metrics` from validating the
        model as saved, if it was validated.
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
                },
            )

        logger.info(f"Saved checkpoint to '{path}'.")

    def _iterate_data(
        self,
        data_iter: Iterator[CityscapesDatapoint],
        datapoint_count: int,
        batch_size: int,
    ) -> TensorDatapointIter:
        """
        Yield batches from `data_iter` until `datapoint_count` datapoints yielded.
        """

        if datapoint_count <= 0:
            return

        remaining = datapoint_count
        iterator = prefetch(
            batch(itertools.islice(data_iter, datapoint_count), batch_size)
        )

        while True:
            with profiler.phase("load raw data"):
                next_batch = next(iterator, None)

            if next_batch is None:
                break

            image_batch, label_batch = next_batch

            remaining -= len(image_batch)
            yield image_batch, label_batch
            if remaining == 0:
                return

        raise ValueError(
            f"Data ran out after {(datapoint_count - remaining):,} datapoints. "
            f"Requested {datapoint_count:,}."
        )

    def _iterate_train_data(self, datapoint_count: int) -> TensorDatapointIter:
        return self._iterate_data(
            self.train_data_iter, datapoint_count, self.train_batch_size
        )

    def _iterate_val_data(self, datapoint_count: int) -> TensorDatapointIter:
        return self._iterate_data(
            self.val_data_iter, datapoint_count, self.val_batch_size
        )

    def _to_tensors(
        self,
        image_batch: torch.Tensor,
        label_batch: torch.Tensor,
        device: torch.device,
        downscale: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert uint8 batches to model-ready tensors on device.

        Images & labels are downscaled by `downscale`.
        """

        # uint8 (B, H, W) --> int64 (B, H / dscale, W / dscale).
        labels = label_batch.to(device, non_blocking=True)
        labels = labels[:, ::downscale, ::downscale]

        if self.label_map.device != device:
            self.label_map = self.label_map.to(device)
        labels = self.label_map[labels.int()]

        # uint8 (B, H, W, 3) --> float (B, 3, H / dscale, W / dscale) in [0, 1].
        images = image_batch.to(device, non_blocking=True)
        images = images.permute(0, 3, 1, 2).float() / MAX_PIXEL_INT_VALUE
        if downscale != 1:
            images = F.interpolate(
                images,
                size=labels.shape[-2:],
                mode="bilinear",
                antialias=True,
            )

        return images, labels

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits, labels, ignore_index=self.label_metadata.ignore_id
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
            loss = self._loss(logits, labels)

        return images, labels, logits, loss

    def _warm_up_step(
        self,
        model: nn.Module,
        image_batch: torch.Tensor,
        label_batch: torch.Tensor,
        device: torch.device,
    ) -> None:
        """
        Evaluate on a batch, do the backward pass, but do not update model.
        """

        images, labels = self._to_tensors(
            image_batch, label_batch, device, downscale=self.train_downscale
        )
        with autocast(device, self.mixed_precision):
            loss = self._loss(model(images), labels)

        loss.backward()
        model.zero_grad(set_to_none=True)

    def _train_step(
        self,
        model: nn.Module,
        image_batch: torch.Tensor,
        label_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate on a batch, and update the model.
        """

        images, labels = self._to_tensors(
            image_batch, label_batch, device, downscale=self.train_downscale
        )
        with autocast(device, self.mixed_precision):
            logits = model(images)
            loss = self._loss(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(optimizer)
        self.grad_scaler.update()

        return images, labels, logits.detach(), loss.detach()

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
        loss_sum = torch.zeros((), device=device)
        confusion = get_empty_confusion_matrix(self.label_metadata, device)

        try:
            model.eval()

            for image_batch, label_batch in self._iterate_val_data(
                datapoint_count
            ):
                is_logging_step = datapoints_seen == 0

                with profiler.phase("infer on val batch", on_gpu=True):
                    images, labels, logits, loss = self._val_step(
                        model, image_batch, label_batch, device
                    )

                    datapoints_seen += len(images)
                    loss_sum += loss * len(images)
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
            self.training_logger.log_val_metrics(
                datapoints_seen=log_train_datapoints_seen,
                total_datapoint_count=log_train_total_datapoint_count,
                loss=(loss_sum / datapoints_seen).item(),
                metrics=metrics,
            )
        return metrics

    def warm_up(self, model: nn.Module) -> None:
        """
        Prepare a model for training.
        """

        device = next(model.parameters()).device
        was_training = model.training

        # Label with an eval class, so the loss is defined.
        label_id = min(self.label_metadata.eval_class_ids)
        train_batch = _get_synthetic_batch(
            self.train_batch_size, self.image_height, self.image_width, label_id
        )
        val_batch = _get_synthetic_batch(
            self.val_batch_size, self.image_height, self.image_width, label_id
        )

        try:
            model.train()
            self._warm_up_step(model, *train_batch, device)

            model.eval()
            self._val_step(model, *val_batch, device)
        finally:
            model.train(was_training)

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
            time_of_last_log = time.perf_counter()

            with torch_profile() as end_torch_profile_step:
                for image_batch, label_batch in self._iterate_train_data(
                    train_datapoint_count
                ):
                    with profiler.phase("infer on train batch", on_gpu=True):
                        images, labels, logits, loss = self._train_step(
                            model,
                            image_batch,
                            label_batch,
                            optimizer,
                            device,
                        )

                        datapoints_seen += len(images)
                        loss_sum_since_last_log += loss * len(images)

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
                        self.training_logger.log_images(
                            log_base_name="train",
                            datapoints_seen=datapoints_seen,
                            images=images,
                            labels=labels,
                            logits=logits,
                            num_images_to_log=log_image_count,
                            image_downscale=(
                                DEFAULT_LOG_IMAGE_DOWNSCALE
                                // self.train_downscale
                            ),
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

                    loss_sum_since_last_log.zero_()

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


@dataclasses.dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    """
    Relevant settings of a training.
    """

    dataset_class: type[CityscapesLabeledDataset]
    train_datapoint_count: int
    val_datapoint_count: int | None
    train_batch_size: int
    val_batch_size: int
    log_every_n_datapoints: int

    init_checkpoint_path: Path | None = None

    base_height: int | None = None
    base_width: int | None = None
    base_channel_count: int | None = None
    level_count: int | None = None

    learning_rate: float = 1e-4

    log_dir: Path | None = None

    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        assert self.log_every_n_datapoints % self.train_batch_size == 0, (
            f"{self.train_batch_size=} must divide "
            f"{self.log_every_n_datapoints=}."
        )

    def to_dict(self) -> dict[str, Any]:
        ret = dataclasses.asdict(self)

        ret["dataset_class"] = self.dataset_class.__name__
        for name in ("init_checkpoint_path", "log_dir"):
            if ret[name] is not None:
                ret[name] = str(ret[name])

        return ret


def _get_unet_config(
    init_checkpoint: dict[str, Any] | None,
    init_checkpoint_path: Path | None,
    output_channel_count: int,
    **requested: int | None,
) -> dict[str, int]:
    """
    Get UNet constructor arguments for a new or checkpointed model.

    For a checkpointed model, set `requested` values must match it.
    """

    if init_checkpoint is None:
        defaults = {
            "base_height": DEFAULT_BASE_HEIGHT,
            "base_width": DEFAULT_BASE_WIDTH,
            "base_channel_count": UNet.DEFAULT_BASE_CHANNEL_COUNT,
            "level_count": UNet.DEFAULT_LEVEL_COUNT,
        }
        return {
            "output_channel_count": output_channel_count,
            **{
                name: value if value is not None else defaults[name]
                for name, value in requested.items()
            },
        }

    config = init_checkpoint["model_config"]
    if config["output_channel_count"] != output_channel_count:
        raise ValueError(
            f"'{init_checkpoint_path}' outputs "
            f"{config['output_channel_count']} classes, but the dataset has "
            f"{output_channel_count}."
        )

    for name, value in requested.items():
        if value is not None and value != config[name]:
            raise ValueError(
                f"{name}={value} conflicts with {name}={config[name]} in "
                f"'{init_checkpoint_path}'."
            )

    return config


def train_unet_on_cityscapes(config: TrainingConfig) -> UNet:
    """
    Train a UNet on finely labeled Cityscapes data, as set by `config`.
    """

    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # Input shapes are fixed, so let cuDNN time & cache the fastest algorithms.
    torch.backends.cudnn.benchmark = True

    dataset = config.dataset_class(seed=config.seed)
    label_metadata = dataset.LABEL_METADATA
    train_data_iter = dataset.iter_fine_train_datapoints(
        num_datapoints=config.train_datapoint_count
    )
    val_data_iter = itertools.chain.from_iterable(
        dataset.iter_fine_val_datapoints(shuffle=True)
        for _ in itertools.count()
    )
    val_datapoint_count = config.val_datapoint_count
    if val_datapoint_count is None:
        val_datapoint_count = dataset.get_fine_val_datapoint_count()

    init_model, init_checkpoint = None, None
    init_checkpoint_path = config.init_checkpoint_path
    checkpoint_metadata = {
        "training_config": config.to_dict(),
        "init_checkpoint": None,
    }
    if init_checkpoint_path is not None:
        init_checkpoint_path = init_checkpoint_path.resolve()
        init_model, init_checkpoint = load_unet_checkpoint(
            init_checkpoint_path, device
        )
        checkpoint_metadata["init_checkpoint"] = get_init_checkpoint_metadata(
            init_checkpoint_path, init_checkpoint
        )

    unet_config = _get_unet_config(
        init_checkpoint,
        init_checkpoint_path,
        output_channel_count=len(label_metadata.class_names),
        base_height=config.base_height,
        base_width=config.base_width,
        base_channel_count=config.base_channel_count,
        level_count=config.level_count,
    )

    with profiler.session():
        with profiler.phase("define model", on_gpu=True):
            if init_model is None:
                model = UNet(**unet_config).to(device)
            else:
                model = init_model
                logger.info(f"Initialized model from '{init_checkpoint_path}'.")

            if os.environ.get("COMPILE_MODEL", "1") != "0":
                model.compile()

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate
            )

        trainer = Trainer(
            image_height=CITYSCAPES_IMAGE_HEIGHT,
            image_width=CITYSCAPES_IMAGE_WIDTH,
            train_data_iter=train_data_iter,
            val_data_iter=val_data_iter,
            train_batch_size=config.train_batch_size,
            val_batch_size=config.val_batch_size,
            label_metadata=label_metadata,
            log_dir=config.log_dir,
            checkpoint_metadata=checkpoint_metadata,
        )

        with profiler.phase("warm up", on_gpu=True):
            trainer.warm_up(model=model)

        trainer.train(
            model=model,
            train_datapoint_count=config.train_datapoint_count,
            val_datapoint_count=val_datapoint_count,
            optimizer=optimizer,
            log_every_n_datapoints=config.log_every_n_datapoints,
        )

    return model


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1] or "/" in sys.argv[1]:
        sys.exit(f"usage: python {sys.argv[0]} label  (label has no '/')")
    label = sys.argv[1]

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    start_time = datetime.now().astimezone()
    train_unet_on_cityscapes(
        TrainingConfig(
            dataset_class=CityscapesPersonLabeledDataset,
            train_datapoint_count=200_000,  # About 67 epochs of the ~2,975 finely labeled training datapoints.
            val_datapoint_count=None,  # Use all 500 labeled datapoints each time.
            train_batch_size=8,
            val_batch_size=1,
            init_checkpoint_path=None,
            log_every_n_datapoints=10_000,
            log_dir=(
                DEFAULT_RUNS_DIR
                / f"{label}_unet_cityscapes_{start_time:%Y%m%d_%H%M%S}"
            ),
        )
    )
