"""
Module to support a random-guess baseline segmentation model.
"""

import torch
from torch import nn

from constants import (
    CITYSCAPES_CLASS_NAMES,
    CITYSCAPES_EVAL_CLASS_IDS,
    DEFAULT_SEED,
)


class RandomSegmenter(nn.Module):
    """
    Randomly predict classes.
    """

    def __init__(self, seed: int = DEFAULT_SEED):
        super().__init__()

        self.num_classes = len(CITYSCAPES_CLASS_NAMES)

        class_ids = sorted(CITYSCAPES_EVAL_CLASS_IDS)
        self.register_buffer(
            "class_ids", torch.tensor(class_ids), persistent=False
        )

        self.generator = torch.Generator().manual_seed(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a (B, 3, H, W) image batch to (B, N, H, W) class logits.
        """

        b, _, h, w = x.shape
        idxs = torch.randint(
            len(self.class_ids), (b, h, w), generator=self.generator
        )
        pred_class_ids = self.class_ids[idxs.to(self.class_ids.device)].to(
            x.device
        )

        logits = torch.zeros(
            (b, self.num_classes, h, w), dtype=x.dtype, device=x.device
        )
        logits.scatter_(1, pred_class_ids.unsqueeze(1), 1.0)

        return logits
