"""
Module to support defining losses for semantic segmentation models.
"""

import torch
import torch.nn.functional as F


def cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    """
    Plain cross entropy.

    Parameters
    ----------
        logits: (B, C, H, W) unnormalized scores, one channel per class.
        labels: (B, H, W) class IDs in [0, C), or ignore_index.
        ignore_index: label value of pixels excluded from the loss.
    """

    return F.cross_entropy(logits, labels, ignore_index=ignore_index)


# Loss name, as in `TrainingConfig.loss_fn_name` --> loss function.
LOSS_FNS = {"cross_entropy": cross_entropy}
