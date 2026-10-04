"""
Module to support logging training state.
"""

import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from semi_sup_seg.constants import MAX_LABEL_COUNT, MAX_PIXEL_INT_VALUE
from semi_sup_seg.data.aug import Augmentation
from semi_sup_seg.data.labels import LabelMetadata

logger = logging.getLogger(__name__)

DEFAULT_LOG_EVERY_N = 500
DEFAULT_LOG_IMAGE_COUNT = 2
DEFAULT_LOG_IMAGE_DOWNSCALE = 2


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

    def log_augmentations(
        self, datapoints_seen: int, augmentation: Augmentation
    ) -> None:
        """
        Log histograms of the parameters of augmentations applied.
        """

        if self.writer is None:
            return

        for name, values in augmentation.to_dict().items():
            self.writer.add_histogram(
                f"train_augmentation/{name}",
                values.float().cpu(),
                datapoints_seen,
            )

    def log_val_metrics(
        self,
        datapoints_seen: int,
        total_datapoint_count: int,
        loss_avg: float,
        loss_stddev: float,
        metrics: dict[str, float],
    ) -> None:
        """
        Log core validation metrics.

        `loss_avg` & `loss_stddev` are over the per-datapoint losses.
        `metrics` comes from `metrics.compute_public_benchmark_metrics`.
        """

        # fmt: off
        logger.info(
            "\n".join(
                [
                    "val logs:",
                    f"\t(train) datapoints seen: {datapoints_seen:,}/{total_datapoint_count:,}",
                    f"\tloss: {loss_avg:.4f} avg, {loss_stddev:.4f} std. dev.",
                    f"\taccuracy: {metrics['accuracy']:.2%}",
                    f"\tmean IoU: {metrics['mean_iou']:.2%}",
                ]
            )
        )
        # fmt: on

        if self.writer is not None:
            self.writer.add_scalar("val/loss_avg", loss_avg, datapoints_seen)
            self.writer.add_scalar(
                "val/loss_stddev", loss_stddev, datapoints_seen
            )
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
