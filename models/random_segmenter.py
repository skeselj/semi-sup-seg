"""
Module to support a random-guess baseline segmentation model.
"""

from collections.abc import Iterable

import torch
from torch import nn

from constants import DEFAULT_SEED


class RandomSegmenter(nn.Module):
    """
    Randomly predict classes.
    """

    def __init__(
        self,
        *,
        output_channel_count: int,
        class_ids: Iterable[int],
        seed: int = DEFAULT_SEED,
    ):
        """
        Construct the model.

        Parameters
        ----------
            output_channel_count: number of channels in the model output, one
                per class.
            class_ids: classes to predict, uniformly at random. Each must be in
                [0, output_channel_count).
            seed: seed for the random predictions.
        """

        super().__init__()

        class_ids = sorted(class_ids)
        if not class_ids:
            raise ValueError("class_ids must be non-empty.")
        if not 0 <= class_ids[0] <= class_ids[-1] < output_channel_count:
            raise ValueError(
                f"{class_ids=} must be in [0, {output_channel_count=})."
            )

        self.num_classes = output_channel_count
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
