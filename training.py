"""
Module to support training segmentation models on datasets.

Example usage:
PROFILE_RESOURCES=1 PROFILE_RESOURCES_TORCH=1 python training.py
"""

import hashlib
import itertools
import logging
import math
import os
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import torch
import torch.nn.functional as F
from profiler import profiler, torch_profile
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from constants import (
    CITYSCAPES_CLASS_COLORS,
    CITYSCAPES_CLASS_NAMES,
    CITYSCAPES_EVAL_CLASS_IDS,
    CITYSCAPES_IMAGE_HEIGHT,
    CITYSCAPES_IMAGE_WIDTH,
    DEFAULT_CHECKPOINT_FILE_NAME,
    DEFAULT_RUNS_DIR,
    DEFAULT_SEED,
    MAX_PIXEL_INT_VALUE,
)
from data import (
    CityscapesDatapoint,
    CityscapesLabeledDataset,
    TensorDatapointIter,
    batch,
    prefetch,
)
from inference import CityscapesEvaluator, load_unet_checkpoint
from metrics import get_confusion_matrix
from models import UNet

standard_logger = logging.getLogger(__name__)

DEFAULT_LOG_EVERY_N = 500
DEFAULT_LOG_IMAGE_COUNT = 2
DEFAULT_LOG_IMAGE_DOWNSCALE = 4

# Model input resolution, for U-Nets trained from scratch.
DEFAULT_BASE_HEIGHT = 256
DEFAULT_BASE_WIDTH = 512

# Train on images & labels downscaled by this factor.
DEFAULT_TRAIN_DOWNSCALE = 2

# Run forward passes in fp16 where safe (on CUDA only).
DEFAULT_MIXED_PRECISION = True

# Label for pixels that do not contribute to the loss: non-benchmark classes.
IGNORE_LABEL = 255


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class Logger:
    """
    Class to support logging training state.
    """

    def __init__(self, log_dir: str | None):
        self.writer = SummaryWriter(log_dir) if log_dir is not None else None

        # Label --> RGB in [0, 1]. Unknown labels, e.g. IGNORE_LABEL, are black.
        self.palette = torch.zeros((256, 3))
        for class_id, color in CITYSCAPES_CLASS_COLORS.items():
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

        small_logits = Logger._shrink(logits, height, width, mode="bilinear")
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
        standard_logger.info(
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

        `metrics` comes from `CityscapesEvaluator.compute_public_benchmark_metrics`.
        """

        # fmt: off
        standard_logger.info(
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


class CityscapesTrainer:
    """
    Class to support training a model w.r.t. a dataset.
    """

    def __init__(
        self,
        train_data_iter: Iterator[CityscapesDatapoint],
        val_data_iter: Iterator[CityscapesDatapoint],
        train_batch_size: int,
        val_batch_size: int,
        log_dir: str | None,
        train_downscale: int = DEFAULT_TRAIN_DOWNSCALE,
        mixed_precision: bool = DEFAULT_MIXED_PRECISION,
        checkpoint_metadata: dict[str, Any] | None = None,
    ):
        self.train_data_iter = train_data_iter
        self.val_data_iter = val_data_iter
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.log_dir = log_dir
        self.train_downscale = train_downscale
        self.mixed_precision = mixed_precision
        self.checkpoint_metadata = checkpoint_metadata or {}

        # Scales the loss up before backward, so small fp16 grads don't
        # underflow to 0, then unscales grads before the optimizer step.
        self.grad_scaler = torch.amp.GradScaler(
            "cuda", enabled=mixed_precision and torch.cuda.is_available()
        )

        self.logger = Logger(log_dir=log_dir)

        # Raw label --> training label: non-benchmark classes are ignored.
        self.label_map = torch.full((256,), IGNORE_LABEL, dtype=torch.long)
        for class_id in CITYSCAPES_EVAL_CLASS_IDS:
            self.label_map[class_id] = class_id

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert uint8 batches to model-ready tensors on device.
        """

        return self._to_downscaled_tensors(
            image_batch, label_batch, device, downscale=1
        )

    def _to_downscaled_tensors(
        self,
        image_batch: torch.Tensor,
        label_batch: torch.Tensor,
        device: torch.device,
        downscale: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Like `_to_tensors`, but labels are downscaled by `downscale`.

        Images stay full-res: the model resizes them to its own resolution in
        one step. Ask the model for outputs at the label resolution instead.
        """

        # uint8 (B, H, W, 3) --> float (B, 3, H, W) in [0, 1].
        images = image_batch.to(device, non_blocking=True)
        images = images.permute(0, 3, 1, 2).float() / MAX_PIXEL_INT_VALUE

        # uint8 (B, H, W) --> int64 (B, H / downscale, W / downscale), with
        # IGNORE_LABEL. Taking every n-th pixel is nearest-neighbor resizing,
        # for free; mapping after it touches fewer pixels.
        labels = label_batch.to(device, non_blocking=True)
        labels = labels[:, ::downscale, ::downscale]

        if self.label_map.device != device:
            self.label_map = self.label_map.to(device)
        labels = self.label_map[labels.int()]

        return images, labels

    def _autocast(self, device: torch.device) -> torch.autocast:
        """
        Context in which eligible ops (e.g. convolutions) run in fp16.
        """

        return torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=self.mixed_precision and device.type == "cuda",
        )

    @staticmethod
    def _loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels, ignore_index=IGNORE_LABEL)

    def _infer_on_train_batch(
        self,
        model: nn.Module,
        image_batch: torch.Tensor,
        label_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate `model` on a train batch, and update it with `optimizer`.
        """

        images, labels = self._to_downscaled_tensors(
            image_batch,
            label_batch,
            device,
            downscale=self.train_downscale,
        )
        with self._autocast(device):
            logits = model(images, output_size=labels.shape[-2:])
            loss = self._loss(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(optimizer)
        self.grad_scaler.update()

        return images, labels, logits.detach(), loss.detach()

    @torch.no_grad()
    def _infer_on_val_batch(
        self,
        model: nn.Module,
        image_batch: torch.Tensor,
        label_batch: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate `model` on a val batch; do not update the model.
        """

        images, labels = self._to_tensors(image_batch, label_batch, device)
        with self._autocast(device):
            logits = model(images)
            loss = self._loss(logits, labels)

        return images, labels, logits, loss

    def _save_checkpoint_to_log_dir(
        self, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, DEFAULT_CHECKPOINT_FILE_NAME)

        temporary_path = f"{path}.tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "model_config": getattr(model, "config", None),
                "optimizer": optimizer.state_dict(),
                "grad_scaler": self.grad_scaler.state_dict(),
                "metadata": self.checkpoint_metadata,
            },
            temporary_path,
        )
        os.replace(temporary_path, path)

        standard_logger.info(f"Saved checkpoint to {path!r}.")

    def _save_checkpoint(
        self, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        """
        Save model & optimizer state under `self.log_dir`, if it is set.
        """

        if self.log_dir is None:
            return

        with profiler.phase("save checkpoint"):
            self._save_checkpoint_to_log_dir(model, optimizer)

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
        Validate `model` w.r.t. `datapoint_count` sampled datapoints.
        """

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        datapoints_seen = 0
        loss_sum = torch.zeros((), device=device)
        confusion = torch.zeros(
            (len(CITYSCAPES_CLASS_NAMES), len(CITYSCAPES_CLASS_NAMES)),
            dtype=torch.long,
            device=device,
        )

        try:
            for image_batch, label_batch in self._iterate_val_data(
                datapoint_count
            ):
                is_logging_step = datapoints_seen == 0

                # Standard step.
                with profiler.phase("infer on val batch", on_gpu=True):
                    images, labels, logits, loss = self._infer_on_val_batch(
                        model, image_batch, label_batch, device
                    )

                    datapoints_seen += len(images)
                    loss_sum += loss * len(images)
                    # Ignored pixels land outside the benchmark rows.
                    confusion += get_confusion_matrix(
                        labels.where(labels != IGNORE_LABEL, 0),
                        logits.argmax(dim=1),
                        len(CITYSCAPES_CLASS_NAMES),
                    ).sum(dim=0)

                if not is_logging_step:
                    continue

                # Logging step.
                with profiler.phase("log for val batch"):
                    self.logger.log_images(
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
            metrics = CityscapesEvaluator.compute_public_benchmark_metrics(
                confusion
            )
            self.logger.log_val_metrics(
                datapoints_seen=log_train_datapoints_seen,
                total_datapoint_count=log_train_total_datapoint_count,
                loss=(loss_sum / datapoints_seen).item(),
                metrics=metrics,
            )
        return metrics

    def warm_up(
        self,
        model: nn.Module,
        height: int = CITYSCAPES_IMAGE_HEIGHT,
        width: int = CITYSCAPES_IMAGE_WIDTH,
    ) -> None:
        """
        Run a train & a val step on synthetic batches, without updating `model`.
        """

        device = next(model.parameters()).device
        was_training = model.training

        def synthetic_batch(
            batch_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            image_batch = torch.zeros(
                (batch_size, height, width, 3), dtype=torch.uint8
            )
            # Label 0 is ignored; use a benchmark class so the loss is defined.
            label_batch = torch.full(
                (batch_size, height, width),
                min(CITYSCAPES_EVAL_CLASS_IDS),
                dtype=torch.uint8,
            )
            return image_batch, label_batch

        try:
            # Train mode, with grad: forward & backward, but no optimizer step.
            model.train()
            image_batch, label_batch = synthetic_batch(self.train_batch_size)
            images, labels = self._to_downscaled_tensors(
                image_batch,
                label_batch,
                device,
                downscale=self.train_downscale,
            )
            with self._autocast(device):
                logits = model(images, output_size=labels.shape[-2:])
                loss = self._loss(logits, labels)
            loss.backward()
            model.zero_grad(set_to_none=True)

            # Eval mode, without grad: a separate compiled graph.
            model.eval()
            image_batch, label_batch = synthetic_batch(self.val_batch_size)
            images, labels = self._to_tensors(image_batch, label_batch, device)
            with torch.no_grad(), self._autocast(device):
                self._loss(model(images), labels)
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
        Train `model` w.r.t. `train_datapoint_count` sampled datapoints.
        """

        device = next(model.parameters()).device
        model.train()

        datapoints_seen = 0
        loss_sum_since_last_log = torch.zeros((), device=device)
        time_of_last_log = time.perf_counter()
        datapoints_seen_at_last_log = 0
        datapoint_index_for_next_log = 0

        try:
            with torch_profile() as end_torch_profile_step:
                for image_batch, label_batch in self._iterate_train_data(
                    train_datapoint_count
                ):
                    # Standard step.
                    with profiler.phase("infer on train batch", on_gpu=True):
                        images, labels, logits, loss = (
                            self._infer_on_train_batch(
                                model,
                                image_batch,
                                label_batch,
                                optimizer,
                                device,
                            )
                        )

                        datapoints_seen += len(images)
                        loss_sum_since_last_log += loss * len(images)

                    end_torch_profile_step()

                    if not (
                        datapoints_seen > datapoint_index_for_next_log
                        or datapoints_seen == train_datapoint_count
                    ):
                        continue

                    # Logging & validation step.
                    datapoints_since_last_log = (
                        datapoints_seen - datapoints_seen_at_last_log
                    )
                    mean_loss_since_last_log = (
                        loss_sum_since_last_log / datapoints_since_last_log
                    ).item()
                    time_since_last_log = time.perf_counter() - time_of_last_log

                    with profiler.phase("log for train batch"):
                        self.logger.log_train_metrics(
                            datapoints_seen=datapoints_seen,
                            total_datapoint_count=train_datapoint_count,
                            datapoints_per_second=(
                                datapoints_since_last_log / time_since_last_log
                            ),
                            loss=mean_loss_since_last_log,
                            learning_rate=optimizer.param_groups[0]["lr"],
                        )
                        self.logger.log_images(
                            log_base_name="train",
                            datapoints_seen=datapoints_seen,
                            images=images,
                            labels=labels,
                            logits=logits,
                            num_images_to_log=log_image_count,
                        )

                    self.validate(
                        model=model,
                        datapoint_count=val_datapoint_count,
                        log_train_datapoints_seen=datapoints_seen,
                        log_train_total_datapoint_count=train_datapoint_count,
                        log_image_count=log_image_count,
                    )

                    loss_sum_since_last_log.zero_()
                    time_of_last_log = time.perf_counter()
                    datapoints_seen_at_last_log = datapoints_seen
                    datapoint_index_for_next_log = (
                        math.ceil(datapoints_seen / log_every_n_datapoints)
                        * log_every_n_datapoints
                    )
        finally:
            self.logger.close()
            self._save_checkpoint(model, optimizer)


def _get_unet_config(
    init_checkpoint: dict[str, Any] | None,
    init_checkpoint_path: str | None,
    **requested: int | None,
) -> dict[str, int]:
    """
    Get UNet constructor arguments for a new or checkpointed model.
    """

    if init_checkpoint is None:
        defaults = {
            "base_height": DEFAULT_BASE_HEIGHT,
            "base_width": DEFAULT_BASE_WIDTH,
            "base_channel_count": UNet.DEFAULT_BASE_CHANNEL_COUNT,
            "level_count": UNet.DEFAULT_LEVEL_COUNT,
        }
        return {
            "output_channel_count": len(CITYSCAPES_CLASS_NAMES),
            **{
                name: value if value is not None else defaults[name]
                for name, value in requested.items()
            },
        }

    config = init_checkpoint["model_config"]
    for name, value in requested.items():
        if value is not None and value != config[name]:
            raise ValueError(
                f"{name}={value} conflicts with {name}={config[name]} in "
                f"{init_checkpoint_path!r}."
            )

    return config


def train_unet_on_cityscapes(
    train_datapoint_count: int,
    val_datapoint_count: int | None,
    train_batch_size: int,
    val_batch_size: int,
    log_every_n_datapoints: int,
    init_checkpoint_path: str | None = None,
    base_height: int | None = None,
    base_width: int | None = None,
    base_channel_count: int | None = None,
    level_count: int | None = None,
    learning_rate: float = 1e-4,
    log_dir: str | None = None,
    seed: int = DEFAULT_SEED,
) -> UNet:
    """
    Train a UNet on finely labeled Cityscapes data.
    """

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    standard_logger.info(f"device: {device}")

    # Input shapes are fixed, so let cuDNN time & cache the fastest algorithms.
    torch.backends.cudnn.benchmark = True

    dataset = CityscapesLabeledDataset(seed=seed)
    train_data_iter = dataset.get_fine_label_train_dataset(
        num_datapoints=train_datapoint_count
    )
    val_data_iter = itertools.chain.from_iterable(
        dataset.get_fine_label_val_dataset(shuffle=True)
        for _ in itertools.count()
    )
    if val_datapoint_count is None:
        # Each validation then consumes exactly one pass of `val_data_iter`.
        val_datapoint_count = dataset.get_fine_label_val_datapoint_count()

    init_model, init_checkpoint = None, None
    checkpoint_metadata = {"init_checkpoint": None}
    if init_checkpoint_path is not None:
        init_checkpoint_path = os.path.abspath(init_checkpoint_path)
        init_model, init_checkpoint = load_unet_checkpoint(
            init_checkpoint_path, device
        )
        checkpoint_metadata["init_checkpoint"] = {
            "path": init_checkpoint_path,
            "sha256": _sha256(init_checkpoint_path),
            "metadata": init_checkpoint.get(
                "metadata"
            ),  # Record whole lineage.
        }

    unet_config = _get_unet_config(
        init_checkpoint,
        init_checkpoint_path,
        base_height=base_height,
        base_width=base_width,
        base_channel_count=base_channel_count,
        level_count=level_count,
    )

    with profiler.session():
        with profiler.phase("define model", on_gpu=True):
            if init_model is None:
                model = UNet(**unet_config).to(device)
            else:
                # Compiling below keeps the loaded parameters as-is.
                model = init_model
                standard_logger.info(
                    f"Initialized model from {init_checkpoint_path!r}."
                )

            if os.environ.get("COMPILE_MODEL", "1") != "0":
                model.compile()

            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        trainer = CityscapesTrainer(
            train_data_iter=train_data_iter,
            val_data_iter=val_data_iter,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            log_dir=log_dir,
            checkpoint_metadata=checkpoint_metadata,
        )

        with profiler.phase("warm up", on_gpu=True):
            trainer.warm_up(model=model)

        trainer.train(
            model=model,
            train_datapoint_count=train_datapoint_count,
            val_datapoint_count=val_datapoint_count,
            optimizer=optimizer,
            log_every_n_datapoints=log_every_n_datapoints,
        )

    return model


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    start_time = datetime.now().astimezone()
    train_unet_on_cityscapes(
        train_datapoint_count=200_000,  # About 67 epochs.
        val_datapoint_count=500,
        train_batch_size=4,
        val_batch_size=1,
        init_checkpoint_path="/home/stefan/hdd/projects/semi-sup-seg/logs/runs/unet_cityscapes_20261002_200314/checkpoint.pt",
        log_every_n_datapoints=25_000,
        log_dir=str(
            DEFAULT_RUNS_DIR / f"unet_cityscapes_{start_time:%Y%m%d_%H%M%S}"
        ),
    )
