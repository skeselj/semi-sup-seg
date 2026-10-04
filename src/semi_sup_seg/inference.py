"""
Module to support running inference with segmentation models.
"""

import logging

import torch
from torch import nn

from semi_sup_seg.config import PseudoLabelingConfig
from semi_sup_seg.data.labels import LabelMetadata
from semi_sup_seg.models import UNet
from semi_sup_seg.precision import DEFAULT_MIXED_PRECISION, autocast

logger = logging.getLogger(__name__)


class PseudoLabeler:
    """
    Class to support labeling images with a teacher.
    """

    def __init__(
        self,
        model: UNet,
        label_metadata: LabelMetadata,
        config: PseudoLabelingConfig,
        mixed_precision: bool = DEFAULT_MIXED_PRECISION,
    ):
        self.model = model.eval()
        self.label_metadata = label_metadata
        self.config = config
        self.mixed_precision = mixed_precision

        self.device = next(model.parameters()).device

        # Predicted class --> training label.
        self.label_map = label_metadata.get_train_label_map().to(self.device)

        # Train datapoints seen --> CPU copy of the student's state then.
        self.teacher_candidate_states: dict[int, dict[str, torch.Tensor]] = {}
        # Train datapoints seen when the teacher's state was checkpointed.
        self.teacher_datapoints_seen: int | None = None

    def update_teacher(self, student: nn.Module, datapoints_seen: int) -> None:
        """
        Keep the student's current state, as a candidate to be the teacher.
        """

        self.teacher_candidate_states[datapoints_seen] = {
            name: tensor.detach().to("cpu", copy=True)
            for name, tensor in student.state_dict().items()
        }

        teacher_datapoints_seen = max(
            (
                candidate
                for candidate in self.teacher_candidate_states
                if candidate <= datapoints_seen - self.config.teacher_lag
            ),
            default=None,
        )
        if teacher_datapoints_seen is None:
            return

        # Older states can't be the teacher anymore.
        for candidate in list(self.teacher_candidate_states):
            if candidate < teacher_datapoints_seen:
                del self.teacher_candidate_states[candidate]

        if (
            datapoints_seen
            < self.config.warmup_datapoints_before_pseudo_labeling
            or teacher_datapoints_seen == self.teacher_datapoints_seen
        ):
            return

        self.model.load_state_dict(
            self.teacher_candidate_states[teacher_datapoints_seen]
        )
        self.teacher_datapoints_seen = teacher_datapoints_seen
        logger.info(
            "Teacher is now the model after "
            f"{teacher_datapoints_seen:,} train datapoints."
        )

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

    def label(self, images: torch.Tensor) -> torch.Tensor:
        """
        Map float (B, 3, H, W) images in [0, 1] to (B, H, W) pseudo-labels.

        Only labels with confidence above the teacher's minimum are kept;
        the rest are ignored.
        """

        labels, confidences = self.predict(images)
        is_ignored = confidences <= self.config.teacher_min_confidence

        return labels.masked_fill(is_ignored, self.label_metadata.ignore_id)
