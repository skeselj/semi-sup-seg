"""
Module to support defining losses for semantic segmentation models.
"""

import torch
import torch.nn.functional as F


def cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    datapoint_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cross entropy, averaged over pixels not ignored.

    Parameters
    ----------
        logits: (B, C, H, W) unnormalized scores, one channel per class.
        labels: (B, H, W) class IDs in [0, C), or ignore_index.
        ignore_index: label value of pixels excluded from the loss.
        datapoint_weights: (B,) multipliers of each datapoint's pixel losses.
            None means all 1, i.e. plain cross entropy.
    """

    if datapoint_weights is None:
        return F.cross_entropy(logits, labels, ignore_index=ignore_index)

    # (B, H, W), with ignored pixels' losses 0.
    pixel_losses = F.cross_entropy(
        logits, labels, ignore_index=ignore_index, reduction="none"
    )
    weighted_loss_sum = (pixel_losses * datapoint_weights[:, None, None]).sum()

    # Divide by the count of pixels not ignored, as plain cross entropy does.
    return weighted_loss_sum / (labels != ignore_index).sum().clamp(min=1)


# Loss name, as in `TrainingConfig.loss_fn_name` --> loss function.
LOSS_FNS = {"cross_entropy": cross_entropy}
