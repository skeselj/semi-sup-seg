"""
Module to support evaluating semantic segmentation models.
"""

import torch


def get_confusion_matrix(
    label: torch.Tensor,
    preds: torch.Tensor,
    num_classes: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Given (B, H, W) label and preds, return the (B, N, N) confusion matrix.

    Element [b, i, j] is num. pixels in image b, with label i, predicted as j.
    """

    assert label.shape == preds.shape
    if not device:
        device = label.device

    batch_size = label.shape[0]
    flat_label = label.reshape(batch_size, -1).long()
    flat_preds = preds.reshape(batch_size, -1).long().to(device)

    batch_idxs = torch.arange(batch_size, device=device).unsqueeze(1)
    # Shape (B, H * W). Each value is b * N**2 + label * N + pred.
    flat_idxs = (
        batch_idxs * num_classes + flat_label
    ) * num_classes + flat_preds

    return torch.bincount(
        flat_idxs.flatten(), minlength=batch_size * num_classes**2
    ).reshape(batch_size, num_classes, num_classes)
