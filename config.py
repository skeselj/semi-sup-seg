"""
Module to support configuring trainings.
"""

import dataclasses
from pathlib import Path
from typing import Any

import losses
from constants import DEFAULT_SEED
from data import (
    CityscapesLabeledDataset,
    CityscapesPersonLabeledDataset,
    CityscapesUnlabeledDataset,
)
from data.aug import AugmentationSampler
from models import UNet

DEFAULT_BASE_HEIGHT = 256
DEFAULT_BASE_WIDTH = 512


@dataclasses.dataclass(frozen=True, kw_only=True)
class PseudoLabelingConfig:
    """
    Settings for pseudo-labeling: using an existing model to make labels.
    """

    # fmt: off
    unlabeled_dataset_class: type[CityscapesUnlabeledDataset] = CityscapesUnlabeledDataset

    warmup_datapoints_before_pseudo_labeling: int = 300_000
    labeled_to_unlabeled_ratio: tuple[int, int] = (1, 7)

    teacher_lag: int = 10_000
    teacher_min_confidence: float = 0.75
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
    labeled_dataset_class: type[CityscapesLabeledDataset]
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
        ret = dataclasses.asdict(self)

        ret["labeled_dataset_class"] = self.labeled_dataset_class.__name__
        if self.pseudo_labeling is not None:
            ret["pseudo_labeling"]["unlabeled_dataset_class"] = (
                self.pseudo_labeling.unlabeled_dataset_class.__name__
            )
        for name in ("init_checkpoint_path", "log_dir"):
            if ret[name] is not None:
                ret[name] = str(ret[name])

        return ret


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
    "naive_supervised_small": TrainingConfig(
        labeled_dataset_class=CityscapesPersonLabeledDataset,
        augmentation_sampler=None,
        pseudo_labeling=None,
        train_datapoint_count=100_000,
        val_datapoint_count=None,
        train_batch_size=8,
        val_batch_size=1,
        init_checkpoint_path=None,
        log_every_n_datapoints=5_000,
    ),
    "augmented_supervised_small": TrainingConfig(
        labeled_dataset_class=CityscapesPersonLabeledDataset,
        augmentation_sampler=AugmentationSampler(),
        pseudo_labeling=None,
        train_datapoint_count=100_000,
        val_datapoint_count=None,
        train_batch_size=8,
        val_batch_size=1,
        init_checkpoint_path=None,
        log_every_n_datapoints=2_000,
    ),
    "augmented_semisupervised_small": TrainingConfig(
        labeled_dataset_class=CityscapesPersonLabeledDataset,
        augmentation_sampler=AugmentationSampler(),
        pseudo_labeling=PseudoLabelingConfig(
            warmup_datapoints_before_pseudo_labeling=10_000,
            labeled_to_unlabeled_ratio=(1, 7),
            teacher_lag=10_000,
        ),
        train_datapoint_count=100_000,
        val_datapoint_count=None,
        train_batch_size=8,
        val_batch_size=1,
        init_checkpoint_path=Path("/mnt/hdd1/projects/semi-sup-seg/logs/runs/augmented_supervised_small_augmented_supervised_small_unet_cityscapes_20261004_115357/checkpoint_100000.pt"),
        log_every_n_datapoints=5_000,
    ),
}
