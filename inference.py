"""
Module to support running inference with segmentation models on datasets.

Example usage:
PROFILE_RESOURCES=1 PROFILE_RESOURCES_TORCH=1 python inference.py /home/stefan/hdd/projects/semi-sup-seg/logs/runs/unet_cityscapes_20261002_200314/checkpoint.pt
"""

import itertools
import logging
import math
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from constants import (
    CITYSCAPES_CLASS_NAMES,
    CITYSCAPES_EVAL_CLASS_IDS,
    MAX_PIXEL_INT_VALUE,
)
from data import CityscapesDatapoint, CityscapesLabeledDataset, prefetch
from metrics import get_confusion_matrix
from models import RandomSegmenter, UNet

standard_logger = logging.getLogger(__name__)

DEFAULT_LOG_EVERY_N = 100


def load_unet_checkpoint(
    path: str, device: torch.device
) -> tuple[UNet, dict[str, Any]]:
    """
    Load a checkpoint saved by training, and the UNet whose weights it holds.
    """

    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_config") is None:
        raise ValueError(f"{path!r} holds no model config.")

    model = UNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])

    return model, checkpoint


def _get_checkpoint_lineage(checkpoint: dict[str, Any]) -> list[str]:
    """
    Get paths of the checkpoints `checkpoint` was initialized from, newest first.
    """

    lineage = []
    metadata = checkpoint.get("metadata")
    while metadata and metadata.get("init_checkpoint"):
        init_checkpoint = metadata["init_checkpoint"]
        lineage.append(init_checkpoint["path"])
        metadata = init_checkpoint.get("metadata")

    return lineage


class Logger:
    """
    Class to support logging inference state.
    """

    def log_metrics(
        self,
        datapoints_seen: int,
        accuracy: float,
        mean_iou: float,
    ) -> None:
        """
        Log core inference metrics.
        """

        # fmt: off
        standard_logger.info(
            "\n".join(
                [
                    "inference logs:",
                    f"\tdatapoints seen: {datapoints_seen:,}",
                    f"\taccuracy: {accuracy:.2%}",
                    f"\tmean IoU: {mean_iou:.2%}",
                ]
            )
        )
        # fmt: on

    def log_class_ious(self, class_to_iou: dict[str, float]) -> None:
        """
        Log a per-class IoU table.
        """

        standard_logger.info(
            "\n".join(
                ["per-class IoU:"]
                + [
                    f"\t{name:15}: {iou:.2%}"
                    for name, iou in class_to_iou.items()
                ]
            )
        )


class CityscapesEvaluator:
    """
    Class to support running inference on a Cityscapes dataset with a model.
    """

    def __init__(self, data_iter: Iterator[CityscapesDatapoint]):
        self.data_iter = data_iter

        self.logger = Logger()

    def _iterate_data(
        self, batch_size: int
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        Yield uint8 (B, H, W, 3) image and (B, H, W) label  batches.
        """

        while datapoints := list(itertools.islice(self.data_iter, batch_size)):
            yield (
                torch.from_numpy(np.stack([dp.image.ary for dp in datapoints])),
                torch.from_numpy(np.stack([dp.label.ary for dp in datapoints])),
            )

    @staticmethod
    def _to_tensors(
        image_batch: torch.Tensor,
        label_batch: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert uint8 batches to model-ready tensors on device.
        """

        # uint8 (B, H, W, 3) --> `dtype` (B, 3, H, W) in [0, 1].
        images = image_batch.to(device, non_blocking=True)
        images = images.permute(0, 3, 1, 2).to(dtype) / MAX_PIXEL_INT_VALUE

        # uint8 (B, H, W) --> int64 (B, H, W).
        labels = label_batch.to(device, non_blocking=True).long()

        return images, labels

    @staticmethod
    def compute_public_benchmark_metrics(
        confusion: torch.Tensor,
    ) -> dict[str, float]:
        """
        Compute benchmark metrics from confusion matrix summed over a dataset.
        """

        eval_ids = sorted(CITYSCAPES_EVAL_CLASS_IDS)

        eval_rows = confusion[eval_ids].double()
        eval_block = eval_rows[:, eval_ids]

        intersection = eval_block.diagonal()
        union = eval_rows.sum(dim=1) + eval_block.sum(dim=0) - intersection
        ious = intersection / union  # NaN for a class never seen

        metrics = {
            "accuracy": (intersection.sum() / eval_rows.sum()).item(),
            "mean_iou": torch.nanmean(ious).item(),
        }
        for class_id, iou in zip(eval_ids, ious.tolist()):
            metrics[f"iou/{CITYSCAPES_CLASS_NAMES[class_id]}"] = iou

        return metrics

    @torch.no_grad()
    def infer(
        self,
        model: nn.Module,
        batch_size: int,
        device: torch.device,
        log_every_n_datapoints: int = DEFAULT_LOG_EVERY_N,
        dtype: torch.dtype = torch.float32,
        mixed_precision: bool = True,
    ) -> dict[str, float]:
        """
        Infer with `model` on self.data_iter.

        If `mixed_precision`, eligible ops run in fp16 (on CUDA only), as in
        training & validation.
        """

        autocast = torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=mixed_precision and device.type == "cuda",
        )

        datapoints_seen = 0
        confusion = torch.zeros(
            (len(CITYSCAPES_CLASS_NAMES), len(CITYSCAPES_CLASS_NAMES)),
            dtype=torch.long,
            device=device,
        )
        datapoint_index_for_next_log = 0
        datapoints_seen_at_last_log = 0

        for image_batch, label_batch in prefetch(
            self._iterate_data(batch_size)
        ):
            # Standard step.
            images, labels = self._to_tensors(
                image_batch, label_batch, dtype, device
            )
            with autocast:
                logits = model(images)  # (B, N, H, W)
            preds = logits.argmax(dim=1)

            datapoints_seen += len(images)
            confusion += get_confusion_matrix(
                labels, preds, len(CITYSCAPES_CLASS_NAMES)
            ).sum(dim=0)

            if datapoints_seen <= datapoint_index_for_next_log:
                continue

            # Logging step.
            metrics = self.compute_public_benchmark_metrics(confusion)
            self.logger.log_metrics(
                datapoints_seen=datapoints_seen,
                accuracy=metrics["accuracy"],
                mean_iou=metrics["mean_iou"],
            )
            datapoints_seen_at_last_log = datapoints_seen

            datapoint_index_for_next_log = (
                math.ceil(datapoints_seen / log_every_n_datapoints)
                * log_every_n_datapoints
            )

        if datapoints_seen == 0:
            raise ValueError("No datapoints to infer on.")

        metrics = self.compute_public_benchmark_metrics(confusion)

        if datapoints_seen != datapoints_seen_at_last_log:
            self.logger.log_metrics(
                datapoints_seen=datapoints_seen,
                accuracy=metrics["accuracy"],
                mean_iou=metrics["mean_iou"],
            )

        self.logger.log_class_ious(
            {
                name.removeprefix("iou/"): value
                for name, value in metrics.items()
                if name.startswith("iou/")
            }
        )

        return metrics


def eval_random_segmenter_on_cityscapes(
    batch_size: int = 4,
    log_every_n_datapoints: int = DEFAULT_LOG_EVERY_N,
) -> dict[str, float]:
    """
    Run RandomSegmenter over all Cityscapes fine-label val datapoints.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    standard_logger.info(f"torch device: {device!r}")

    dataset = CityscapesLabeledDataset()
    evaluator = CityscapesEvaluator(
        data_iter=dataset.get_fine_label_val_dataset()
    )
    model = RandomSegmenter().to(device).eval()

    return evaluator.infer(
        model=model,
        batch_size=batch_size,
        device=device,
        log_every_n_datapoints=log_every_n_datapoints,
    )


def eval_unet_on_cityscapes(
    checkpoint_path: str | None = None,
    batch_size: int = 1,
    log_every_n_datapoints: int = DEFAULT_LOG_EVERY_N,
    mixed_precision: bool = True,
) -> dict[str, float]:
    """
    Run a checkpointed UNet over all Cityscapes fine-label val datapoints.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    standard_logger.info(f"torch device: {device!r}")

    # Input shapes are fixed, so let cuDNN time & cache the fastest algorithms.
    torch.backends.cudnn.benchmark = True

    checkpoint_path = os.path.abspath(checkpoint_path)

    model, checkpoint = load_unet_checkpoint(checkpoint_path, device)
    model.eval()
    lineage = _get_checkpoint_lineage(checkpoint)
    standard_logger.info(
        "\n".join(
            [
                f"Loaded model from {checkpoint_path!r}, initialized from:",
                *([f"\t{path!r}" for path in lineage] or ["\tnothing"]),
            ]
        )
    )

    if os.environ.get("COMPILE_MODEL", "1") != "0":
        model.compile()

    dataset = CityscapesLabeledDataset()
    evaluator = CityscapesEvaluator(
        data_iter=dataset.get_fine_label_val_dataset()
    )

    return evaluator.infer(
        model=model,
        batch_size=batch_size,
        device=device,
        log_every_n_datapoints=log_every_n_datapoints,
        mixed_precision=mixed_precision,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    if len(sys.argv) > 2:
        sys.exit(f"usage: python {sys.argv[0]} [checkpoint_path]")
    eval_unet_on_cityscapes(
        checkpoint_path=sys.argv[1] if len(sys.argv) == 2 else None
    )
