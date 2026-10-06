"""
Module to support running inference with segmentation models.
"""

import dataclasses
import logging

import torch
from torch import nn

from semi_sup_seg.data.labels import LabelMetadata
from semi_sup_seg.models import UNet
from semi_sup_seg.precision import DEFAULT_MIXED_PRECISION, autocast

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PseudoLabelingConfig:
    """
    Settings for pseudo-labeling: using an existing model to make labels.
    """

    # fmt: off
    warmup_datapoints_before_pseudo_labeling: int
    labeled_to_unlabeled_ratio: tuple[int, int]
    labeled_and_unlabeled_loss_multipliers: tuple[float, float]

    teacher_lag: int = 0
    teacher_ema_decay: float | None = 0.9
    teacher_min_confidence: float | None = None
    # fmt: on

    def __post_init__(self) -> None:
        if not (
            0
            <= self.teacher_lag
            <= self.warmup_datapoints_before_pseudo_labeling
        ):
            raise ValueError(
                f"{self.teacher_lag=} must be in "
                f"[0, {self.warmup_datapoints_before_pseudo_labeling=}]."
            )
        if min(self.labeled_and_unlabeled_loss_multipliers) < 0:
            raise ValueError(
                f"{self.labeled_and_unlabeled_loss_multipliers=} must be "
                "non-negative."
            )
        if (
            self.teacher_ema_decay is not None
            and not 0 <= self.teacher_ema_decay < 1
        ):
            raise ValueError(f"{self.teacher_ema_decay=} must be in [0, 1).")
        if (
            self.teacher_min_confidence is not None
            and not 0 <= self.teacher_min_confidence < 1
        ):
            raise ValueError(
                f"{self.teacher_min_confidence=} must be in [0, 1)."
            )


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
        self.label_map = label_metadata.get_filter_to_eval_labels_map().to(
            self.device
        )

        # Train datapoints seen --> CPU copy of the student's state then.
        self.teacher_candidate_states: dict[int, dict[str, torch.Tensor]] = {}
        # Train datapoints seen when student state was last taken into teacher.
        self.teacher_datapoints_seen: int | None = None

    @torch.no_grad()
    def _blend_into_teacher(
        self, state: dict[str, torch.Tensor], decay: float
    ) -> None:
        """
        Set the teacher to `decay * teacher + (1 - decay) * state`, in place.
        """

        for name, tensor in self.model.state_dict().items():
            other = state[name].to(tensor.device)
            if tensor.is_floating_point():
                tensor.mul_(decay).add_(other, alpha=1 - decay)
            else:
                tensor.copy_(other)

    def update_teacher(self, student: nn.Module, datapoints_seen: int) -> None:
        """
        Update the teacher's current state with `student`.
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

        for candidate in list(self.teacher_candidate_states):
            if candidate < teacher_datapoints_seen:
                del self.teacher_candidate_states[candidate]

        if (
            datapoints_seen
            < self.config.warmup_datapoints_before_pseudo_labeling
            or teacher_datapoints_seen == self.teacher_datapoints_seen
        ):
            return

        candidate_state = self.teacher_candidate_states[teacher_datapoints_seen]
        decay = self.config.teacher_ema_decay
        if decay is None or self.teacher_datapoints_seen is None:
            self.model.load_state_dict(candidate_state)
            logger.info(
                f"Taking the {teacher_datapoints_seen:,} datapoint student, "
                f"model, and setting teacher to it."
            )
        else:
            self._blend_into_teacher(candidate_state, decay)
            logger.info(
                f"Taking the {teacher_datapoints_seen:,} datapoint student, "
                f"and incorporating it into teacher with {decay=}."
            )

        self.teacher_datapoints_seen = teacher_datapoints_seen

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

        Only labels with confidence above the teacher's minimum are kept, if
        it has one; the rest are ignored.
        """

        labels, confidences = self.predict(images)
        min_confidence = self.config.teacher_min_confidence
        if min_confidence is None:
            return labels

        is_ignored = confidences <= min_confidence

        return labels.masked_fill(is_ignored, self.label_metadata.ignore_id)
