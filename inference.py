"""
Module to support running inference with segmentation models.
"""

import logging

import torch

from constants import MAX_LABEL_COUNT
from data import LabelMetadata
from models import UNet
from precision import DEFAULT_MIXED_PRECISION, autocast

logger = logging.getLogger(__name__)


class PseudoLabeller:
    """
    Class to support predicting classes for images with an existing model.
    """

    def __init__(
        self,
        model: UNet,
        label_metadata: LabelMetadata,
        mixed_precision: bool = DEFAULT_MIXED_PRECISION,
    ):
        self.model = model.eval()
        self.label_metadata = label_metadata
        self.mixed_precision = mixed_precision

        self.device = next(model.parameters()).device

        # Raw label --> training label.
        self.label_map = torch.full(
            (MAX_LABEL_COUNT,),
            label_metadata.ignore_id,
            dtype=torch.long,
            device=self.device,
        )
        for class_id in label_metadata.eval_class_ids:
            self.label_map[class_id] = class_id

    @torch.no_grad()
    def predict_logits(self, images: torch.Tensor) -> torch.Tensor:
        """
        Map float (B, 3, H, W) images in [0, 1] to (B, N, H, W) logits.
        """

        with autocast(self.device, self.mixed_precision):
            return self.model(images)

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Map float (B, 3, H, W) images in [0, 1] to labels & confidences.
        """

        probs = self.predict_logits(images).float().softmax(dim=1)
        confidences, classes = probs.max(dim=1)

        return self.label_map[classes], confidences
