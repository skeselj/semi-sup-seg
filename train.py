"""
Script to train segmentation models on datasets.

Example usage:
PROFILE_RESOURCES=1 PROFILE_RESOURCES_TORCH=1 python train.py $PRESET $LABEL
"""

import dataclasses
import itertools
import logging
import os
import sys
from datetime import datetime

import torch
from profiler import profiler

from semi_sup_seg import losses
from semi_sup_seg.checkpoints import (
    get_init_checkpoint_metadata,
    load_unet_checkpoint,
)
from semi_sup_seg.config import (
    TRAINING_PRESETS,
    TrainingConfig,
    get_unet_config,
)
from semi_sup_seg.constants import (
    DEFAULT_RUNS_DIR,
)
from semi_sup_seg.inference import PseudoLabeler
from semi_sup_seg.models import UNet
from semi_sup_seg.trainer import Trainer

logger = logging.getLogger(__name__)


def train_unet_on_cityscapes(config: TrainingConfig) -> UNet:
    """
    Train a UNet on Cityscapes data, as set by `config`.
    """

    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # Input shapes are fixed, so let cuDNN time & cache the fastest algorithms.
    torch.backends.cudnn.benchmark = True

    labeled_data = config.labeled_data
    labeled_dataset = labeled_data.dataset_class(
        image_dir=labeled_data.image_dir,
        fine_label_dir=labeled_data.fine_label_dir,
        seed=config.seed,
    )
    label_metadata = labeled_dataset.LABEL_METADATA
    labeled_train_data_iter = labeled_dataset.iter_fine_train_datapoints(
        num_datapoints=config.train_datapoint_count
    )
    labeled_val_data_iter = itertools.chain.from_iterable(
        labeled_dataset.iter_fine_val_datapoints(shuffle=True)
        for _ in itertools.count()
    )
    val_datapoint_count = config.val_datapoint_count
    if val_datapoint_count is None:
        val_datapoint_count = labeled_dataset.get_fine_val_datapoint_count()

    init_model, init_checkpoint = None, None
    init_checkpoint_path = config.init_checkpoint_path
    checkpoint_metadata = {
        "training_config": config.to_dict(),
        "init_checkpoint": None,
    }
    if init_checkpoint_path is not None:
        init_checkpoint_path = init_checkpoint_path.resolve()
        init_model, init_checkpoint = load_unet_checkpoint(
            init_checkpoint_path, device
        )
        checkpoint_metadata["init_checkpoint"] = get_init_checkpoint_metadata(
            init_checkpoint_path, init_checkpoint
        )

    unet_config = get_unet_config(
        init_checkpoint,
        init_checkpoint_path,
        output_channel_count=len(label_metadata.class_names),
        base_height=config.base_height,
        base_width=config.base_width,
        base_channel_count=config.base_channel_count,
        level_count=config.level_count,
    )

    with profiler.session():
        with profiler.phase("define model", on_gpu=True):
            if init_model is None:
                model = UNet(**unet_config).to(device)
            else:
                model = init_model
                logger.info(f"Initialized model from '{init_checkpoint_path}'.")

            if os.environ.get("COMPILE_MODEL", "1") != "0":
                model.compile()

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate
            )

            pseudo_labeling = config.pseudo_labeling
            pseudo_labeler, unlabeled_train_data_iter = None, None
            unlabeled_image_size = None
            if pseudo_labeling is not None:
                teacher = UNet(**unet_config).to(device)
                if os.environ.get("COMPILE_MODEL", "1") != "0":
                    teacher.compile()
                pseudo_labeler = PseudoLabeler(
                    teacher, label_metadata, pseudo_labeling
                )

                unlabeled_dataset = pseudo_labeling.unlabeled_dataset_class(
                    image_dir=pseudo_labeling.unlabeled_image_dir,
                    seed=config.seed,
                    keep_every_nth_frame=(
                        pseudo_labeling.unlabeled_keep_every_nth_frame
                    ),
                )
                unlabeled_image_size = unlabeled_dataset.get_image_size()
                unlabeled_train_data_iter = (
                    unlabeled_dataset.iter_train_datapoints(
                        num_datapoints=config.train_datapoint_count
                    )
                )

        trainer = Trainer(
            labeled_image_size=labeled_dataset.get_image_size(),
            labeled_train_data_iter=labeled_train_data_iter,
            labeled_val_data_iter=labeled_val_data_iter,
            train_batch_size=config.train_batch_size,
            val_batch_size=config.val_batch_size,
            label_metadata=label_metadata,
            log_dir=config.log_dir,
            loss_fn=losses.LOSS_FNS[config.loss_fn_name],
            augmentation_sampler=config.augmentation_sampler,
            checkpoint_metadata=checkpoint_metadata,
            pseudo_labeler=pseudo_labeler,
            unlabeled_train_data_iter=unlabeled_train_data_iter,
            unlabeled_image_size=unlabeled_image_size,
        )

        with profiler.phase("warm up", on_gpu=True):
            trainer.warm_up(model=model)

        trainer.train(
            model=model,
            train_datapoint_count=config.train_datapoint_count,
            val_datapoint_count=val_datapoint_count,
            optimizer=optimizer,
            log_every_n_datapoints=config.log_every_n_datapoints,
        )

    return model


if __name__ == "__main__":
    if (
        len(sys.argv) != 3
        or sys.argv[1] not in TRAINING_PRESETS
        or not sys.argv[2]
        or "/" in sys.argv[2]
    ):
        sys.exit(
            f"usage: python {sys.argv[0]} preset label  (label has no '/')\n"
            f"presets: {', '.join(TRAINING_PRESETS)}"
        )
    preset, label = sys.argv[1:]

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    start_time = datetime.now().astimezone()
    train_unet_on_cityscapes(
        dataclasses.replace(
            TRAINING_PRESETS[preset],
            log_dir=(
                DEFAULT_RUNS_DIR
                / f"{label}_{preset}_unet_cityscapes_{start_time:%Y%m%d_%H%M%S}"
            ),
        )
    )
