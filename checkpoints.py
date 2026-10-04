"""
Module to support saving & loading model checkpoints.
"""

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from models import UNet


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def get_checkpoint_file_name(
    datapoints_seen: int, total_datapoint_count: int
) -> str:
    """
    Get the file name of a checkpoint saved after `datapoints_seen` of
    `total_datapoint_count` train datapoints.

    The count is zero-padded to the width of the total, so file names sort in
    training order.
    """

    width = len(str(total_datapoint_count))
    return f"checkpoint_{datapoints_seen:0{width}d}.pt"


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_scaler: torch.amp.GradScaler,
    metadata: dict[str, Any],
) -> None:
    """
    Save a checkpoint to `path`, atomically: a crash never leaves a partial one.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(f"{path.name}.tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": getattr(model, "config", None),
            "optimizer": optimizer.state_dict(),
            "grad_scaler": grad_scaler.state_dict(),
            "metadata": metadata,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def load_unet_checkpoint(
    path: Path, device: torch.device
) -> tuple[UNet, dict[str, Any]]:
    """
    Load a checkpoint saved by training, and the UNet whose weights it holds.
    """

    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_config") is None:
        raise ValueError(f"'{path}' holds no model config.")

    model = UNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])

    return model, checkpoint


def get_init_checkpoint_metadata(
    path: Path, checkpoint: dict[str, Any]
) -> dict[str, Any]:
    """
    Describe `checkpoint`, loaded from `path`, for a model initialized from it.

    Its own metadata is included, so the whole lineage is recorded.
    """

    return {
        "path": str(path),
        "sha256": _sha256(path),
        "metadata": checkpoint.get("metadata"),
    }


def get_checkpoint_lineage(checkpoint: dict[str, Any]) -> list[str]:
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


def get_checkpoint_dataset_name(checkpoint: dict[str, Any]) -> str:
    """
    Get the class name of the dataset `checkpoint` was trained on.
    """

    return checkpoint["metadata"]["training_config"]["labeled_dataset_class"]
