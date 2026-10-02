import torch
from torch import nn

from constants import (
    CITYSCAPES_CLASS_NAMES,
    CITYSCAPES_VOID_CLASS_IDS,
    DEFAULT_SEED,
)


class RandomSegmenter(nn.Module):
    """
    Predict a uniformly random non-void class for every pixel.
    """

    def __init__(self, seed: int = DEFAULT_SEED):
        super().__init__()

        self.num_classes = len(CITYSCAPES_CLASS_NAMES)

        class_ids = sorted(
            set(CITYSCAPES_CLASS_NAMES) - set(CITYSCAPES_VOID_CLASS_IDS)
        )
        # Moves with .to(device); derived from constants, so not saved.
        self.register_buffer(
            "class_ids", torch.tensor(class_ids), persistent=False
        )

        # Sampled on the CPU so the same seed gives the same output on any
        # device.
        self.generator = torch.Generator().manual_seed(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a (B, 3, H, W) image batch to (B, N, H, W) class logits.

        The logits are one-hot: 1 for a uniformly random non-void class at
        each pixel, 0 elsewhere. That takes one random draw per pixel,
        instead of one per pixel per class.
        """

        b, _, h, w = x.shape
        idxs = torch.randint(
            len(self.class_ids), (b, h, w), generator=self.generator
        )
        pred_class_ids = self.class_ids[idxs.to(self.class_ids.device)]

        # Scatter into a float tensor, rather than F.one_hot, which would
        # build an int64 (B, H, W, N) tensor first: 2x the memory.
        logits = torch.zeros(
            (b, self.num_classes, h, w), dtype=x.dtype, device=x.device
        )
        logits.scatter_(1, pred_class_ids.unsqueeze(1).to(x.device), 1.0)
        return logits
