"""
Module to support configuring trainings.
"""

import dataclasses
from pathlib import Path
from typing import Any

from semi_sup_seg import losses
from semi_sup_seg.constants import CITYSCAPES_DIR, DEFAULT_SEED
from semi_sup_seg.data import (
    CityscapesLabeledDataset,
    CityscapesPersonLabeledDataset,
    CityscapesUnlabeledDataset,
)
from semi_sup_seg.data.aug import AugmentationSampler
from semi_sup_seg.models import UNet

DEFAULT_BASE_HEIGHT = 256
DEFAULT_BASE_WIDTH = 512


def _to_plain(value: Any) -> Any:
    """
    Convert classes to their names & paths to strings, recursively.
    """

    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, Path):
        return str(value)
    return value


@dataclasses.dataclass(frozen=True, kw_only=True)
class LabeledDataConfig:
    """
    Settings for the labeled data: where it is.
    """

    # fmt: off
    dataset_class: type[CityscapesLabeledDataset] = CityscapesLabeledDataset
    image_dir: Path = CITYSCAPES_DIR / "leftImg8bit_2x_downsampled"
    fine_label_dir: Path = CITYSCAPES_DIR / "gtFine_2x_downsampled"
    # fmt: on


@dataclasses.dataclass(frozen=True, kw_only=True)
class PseudoLabelingConfig:
    """
    Settings for pseudo-labeling: using an existing model to make labels.
    """

    # fmt: off
    unlabeled_dataset_class: type[CityscapesUnlabeledDataset] = CityscapesUnlabeledDataset
    unlabeled_image_dir: Path = CITYSCAPES_DIR / "leftImg8bit_sequence_2x_downsampled"
    unlabeled_keep_every_nth_frame: int = 2

    warmup_datapoints_before_pseudo_labeling: int = 300_000
    labeled_to_unlabeled_ratio: tuple[int, int] = (1, 7)
    labeled_and_unlabeled_loss_multipliers: tuple[float, float] = (4/1, 4/7)

    teacher_lag: int = 10_000
    teacher_min_confidence: float = 0.60
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
        if not 0 <= self.teacher_min_confidence < 1:
            raise ValueError(
                f"{self.teacher_min_confidence=} must be in [0, 1)."
            )


@dataclasses.dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    """
    Settings for training a model.
    """

    # fmt: off
    labeled_data: LabeledDataConfig
    pseudo_labeling: PseudoLabelingConfig | None = None
    augmentation_sampler: AugmentationSampler | None = None

    train_datapoint_count: int
    val_datapoint_count: int | None
    train_batch_size: int
    val_batch_size: int

    init_checkpoint_path: Path | None = None

    base_height: int | None = None
    base_width: int | None = None
    base_channel_count: int | None = None
    level_count: int | None = None

    loss_fn_name: str = "cross_entropy"
    learning_rate: float = 1e-4

    log_dir: Path | None = None
    log_every_n_datapoints: int

    seed: int = DEFAULT_SEED
    # fmt: on

    def __post_init__(self) -> None:
        if self.log_every_n_datapoints % self.train_batch_size != 0:
            raise ValueError(
                f"{self.train_batch_size=} must divide "
                f"{self.log_every_n_datapoints=}."
            )
        if self.loss_fn_name not in losses.LOSS_FNS:
            raise ValueError(
                f"{self.loss_fn_name=} must be one of {list(losses.LOSS_FNS)}."
            )

        pseudo_labeling = self.pseudo_labeling
        if pseudo_labeling is None:
            return

        # Checks that involve both the pseudo-labeling & other settings.
        ratio_sum = sum(pseudo_labeling.labeled_to_unlabeled_ratio)
        if self.train_batch_size % ratio_sum != 0:
            raise ValueError(
                f"{ratio_sum=} of labeled_to_unlabeled_ratio must divide "
                f"{self.train_batch_size=}."
            )
        if self.train_datapoint_count % self.train_batch_size != 0:
            raise ValueError(
                f"{self.train_batch_size=} must divide "
                f"{self.train_datapoint_count=}."
            )
        for name in ("warmup_datapoints_before_pseudo_labeling", "teacher_lag"):
            value = getattr(pseudo_labeling, name)
            if value % self.log_every_n_datapoints != 0:
                raise ValueError(
                    f"{self.log_every_n_datapoints=} must divide "
                    f"{name}={value}."
                )

    def to_dict(self) -> dict[str, Any]:
        """
        Get these settings as plain values, e.g. classes as their names.
        """

        return _to_plain(dataclasses.asdict(self))


def get_unet_config(
    init_checkpoint: dict[str, Any] | None,
    init_checkpoint_path: Path | None,
    output_channel_count: int,
    **requested: int | None,
) -> dict[str, int]:
    """
    Get UNet constructor arguments for a new or checkpointed model.

    For a checkpointed model, set `requested` values must match it.
    """

    if init_checkpoint is None:
        defaults = {
            "base_height": DEFAULT_BASE_HEIGHT,
            "base_width": DEFAULT_BASE_WIDTH,
            "base_channel_count": UNet.DEFAULT_BASE_CHANNEL_COUNT,
            "level_count": UNet.DEFAULT_LEVEL_COUNT,
        }
        return {
            "output_channel_count": output_channel_count,
            **{
                name: value if value is not None else defaults[name]
                for name, value in requested.items()
            },
        }

    config = init_checkpoint["model_config"]
    if config["output_channel_count"] != output_channel_count:
        raise ValueError(
            f"'{init_checkpoint_path}' outputs "
            f"{config['output_channel_count']} classes, but the dataset has "
            f"{output_channel_count}."
        )

    for name, value in requested.items():
        if value is not None and value != config[name]:
            raise ValueError(
                f"{name}={value} conflicts with {name}={config[name]} in "
                f"'{init_checkpoint_path}'."
            )

    return config


# Preset name --> settings of that kind of training, except where it logs.
TRAINING_PRESETS: dict[str, TrainingConfig] = {
    "plain_supervised": TrainingConfig(
        labeled_data=LabeledDataConfig(
            dataset_class=CityscapesPersonLabeledDataset
        ),
        augmentation_sampler=None,
        pseudo_labeling=None,
        train_datapoint_count=250_000,
        val_datapoint_count=None,
        train_batch_size=8,
        val_batch_size=1,
        init_checkpoint_path=None,
        learning_rate=1e-4,
        log_every_n_datapoints=5_000,
    ),
    "augmented_supervised": TrainingConfig(
        labeled_data=LabeledDataConfig(
            dataset_class=CityscapesPersonLabeledDataset
        ),
        augmentation_sampler=AugmentationSampler(),
        pseudo_labeling=None,
        train_datapoint_count=1_000_000,
        val_datapoint_count=None,
        train_batch_size=8,
        val_batch_size=1,
        init_checkpoint_path=None,
        learning_rate=1e-4,
        log_every_n_datapoints=10_000,
    ),
    "augmented_semisupervised": TrainingConfig(
        labeled_data=LabeledDataConfig(
            dataset_class=CityscapesPersonLabeledDataset
        ),
        augmentation_sampler=AugmentationSampler(),
        pseudo_labeling=PseudoLabelingConfig(
            warmup_datapoints_before_pseudo_labeling=50_000,
            labeled_to_unlabeled_ratio=(1, 7),
            labeled_and_unlabeled_loss_multipliers=(4 / 1, 4 / 7),
            teacher_lag=50_000,
            teacher_min_confidence=0.60,
        ),
        train_datapoint_count=3_000_000,
        val_datapoint_count=None,
        train_batch_size=8,
        val_batch_size=1,
        init_checkpoint_path=Path("./logs/runs/oct_04_evening_augmented_supervised_unet_cityscapes_20261004_215123/checkpoint_0950000.pt"),
        learning_rate=1e-5,
        log_every_n_datapoints=10_000,
    ),
}
