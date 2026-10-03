"""
Module to support running inference with segmentation models on datasets.

Example usage:
PROFILE_RESOURCES=1 PROFILE_RESOURCES_TORCH=1 python inference.py
"""

import logging
import math
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import nn

from checkpoints import (
    get_checkpoint_dataset_name,
    get_checkpoint_lineage,
    load_unet_checkpoint,
)
from constants import MAX_PIXEL_INT_VALUE
from data import (
    CITYSCAPES_DATASET_CLASSES,
    CityscapesDatapoint,
    CityscapesLabeledDataset,
    LabelMetadata,
)
from data_iter import batch, prefetch
from metrics import (
    compute_public_benchmark_metrics,
    get_batch_confusion_matrix,
    get_empty_confusion_matrix,
)
from models import RandomSegmenter
from precision import DEFAULT_MIXED_PRECISION, autocast

logger = logging.getLogger(__name__)

DEFAULT_LOG_EVERY_N = 100


class InferenceLogger:
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
        logger.info(
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

        logger.info(
            "\n".join(
                ["per-class IoU:"]
                + [
                    f"\t{name:15}: {iou:.2%}"
                    for name, iou in class_to_iou.items()
                ]
            )
        )


class Evaluator:
    """
    Class to support running inference on a dataset with a model.
    """

    def __init__(
        self,
        data_iter: Iterator[CityscapesDatapoint],
        label_metadata: LabelMetadata,
    ):
        self.data_iter = data_iter
        self.label_metadata = label_metadata

        self.inference_logger = InferenceLogger()

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

    @torch.no_grad()
    def infer(
        self,
        model: nn.Module,
        batch_size: int,
        device: torch.device,
        log_every_n_datapoints: int = DEFAULT_LOG_EVERY_N,
        dtype: torch.dtype = torch.float32,
        mixed_precision: bool = DEFAULT_MIXED_PRECISION,
    ) -> dict[str, float]:
        """
        Infer with `model` on self.data_iter.
        """

        datapoints_seen = 0
        confusion = get_empty_confusion_matrix(self.label_metadata, device)
        datapoint_index_for_next_log = 0
        datapoints_seen_at_last_log = 0

        for image_batch, label_batch in prefetch(
            batch(self.data_iter, batch_size)
        ):
            # Standard step.
            images, labels = self._to_tensors(
                image_batch, label_batch, dtype, device
            )
            with autocast(device, mixed_precision):
                logits = model(images)  # (B, N, H, W)
            preds = logits.argmax(dim=1)

            datapoints_seen += len(images)
            confusion += get_batch_confusion_matrix(
                labels, preds, self.label_metadata
            )

            if datapoints_seen <= datapoint_index_for_next_log:
                continue

            # Logging step.
            metrics = compute_public_benchmark_metrics(
                confusion, self.label_metadata
            )
            self.inference_logger.log_metrics(
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

        metrics = compute_public_benchmark_metrics(
            confusion, self.label_metadata
        )

        if datapoints_seen != datapoints_seen_at_last_log:
            self.inference_logger.log_metrics(
                datapoints_seen=datapoints_seen,
                accuracy=metrics["accuracy"],
                mean_iou=metrics["mean_iou"],
            )

        self.inference_logger.log_class_ious(
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
    logger.info(f"torch device: {device!r}")

    dataset = CityscapesLabeledDataset()
    evaluator = Evaluator(
        data_iter=dataset.iter_fine_val_datapoints(),
        label_metadata=dataset.LABEL_METADATA,
    )
    model = (
        RandomSegmenter(
            output_channel_count=len(dataset.LABEL_METADATA.class_names),
            class_ids=dataset.LABEL_METADATA.eval_class_ids,
        )
        .to(device)
        .eval()
    )

    return evaluator.infer(
        model=model,
        batch_size=batch_size,
        device=device,
        log_every_n_datapoints=log_every_n_datapoints,
    )


def eval_unet_on_cityscapes(
    checkpoint_path: Path,
    batch_size: int = 1,
    log_every_n_datapoints: int = DEFAULT_LOG_EVERY_N,
    mixed_precision: bool = DEFAULT_MIXED_PRECISION,
) -> dict[str, float]:
    """
    Run a checkpointed UNet over all Cityscapes fine-label val datapoints.

    The val datapoints come from the dataset the UNet was trained on.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"torch device: {device!r}")

    # Input shapes are fixed, so let cuDNN time & cache the fastest algorithms.
    torch.backends.cudnn.benchmark = True

    checkpoint_path = checkpoint_path.resolve()

    model, checkpoint = load_unet_checkpoint(checkpoint_path, device)
    model.eval()
    lineage = get_checkpoint_lineage(checkpoint)
    logger.info(
        "\n".join(
            [
                f"Loaded model from '{checkpoint_path}', initialized from:",
                *([f"\t'{path}'" for path in lineage] or ["\tnothing"]),
            ]
        )
    )

    dataset_name = get_checkpoint_dataset_name(checkpoint)
    dataset_class = CITYSCAPES_DATASET_CLASSES[dataset_name]
    logger.info(f"Evaluating on {dataset_name}.")

    if os.environ.get("COMPILE_MODEL", "1") != "0":
        model.compile()

    dataset = dataset_class()
    evaluator = Evaluator(
        data_iter=dataset.iter_fine_val_datapoints(),
        label_metadata=dataset.LABEL_METADATA,
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
    if len(sys.argv) != 2:
        sys.exit(f"usage: python {sys.argv[0]} checkpoint_path")
    eval_unet_on_cityscapes(checkpoint_path=Path(sys.argv[1]))
