"""
Module to support accessing Cityscapes data.
"""

import collections
import dataclasses
import logging
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from semi_sup_seg.constants import (
    CITYSCAPES_CLASS_COLORS,
    CITYSCAPES_CLASS_NAMES,
    CITYSCAPES_DIR,
    CITYSCAPES_EVAL_CLASSES,
    CITYSCAPES_PERSON_CLASSES,
    CITYSCAPES_VOID_CLASSES,
    DEFAULT_SEED,
    IGNORE_LABEL_ID,
)
from semi_sup_seg.data.datapoints import Datapoint
from semi_sup_seg.data.datasets import SplitDataset
from semi_sup_seg.data.labels import IGNORE_COLOR, LabelMetadata

logger = logging.getLogger(__name__)

# Cityscapes file names are <city>_<clip>_<frame>_<suffix>.
_IMAGE_NAME_PATTERN = re.compile(
    r"^(?P<city>[a-z-]+)_(?P<clip_idx>\d{6})_(?P<frame_idx>\d{6})"
    r"_leftImg8bit\.png$"
)
_LABEL_NAME_PATTERN = re.compile(
    r"^(?P<city>[a-z-]+)_(?P<clip_idx>\d{6})_(?P<frame_idx>\d{6})"
    r"_gt(?:Fine|Coarse)_labelIds\.png$"
)


@dataclasses.dataclass
class CityscapesDatapoint(Datapoint):
    """
    A Cityscapes datapoint, with what's parsed from its image's file name.
    """

    city: str = dataclasses.field(init=False)
    clip_idx: str = dataclasses.field(init=False)
    frame_idx: str = dataclasses.field(init=False)

    def __post_init__(self):
        m = _IMAGE_NAME_PATTERN.match(self.image_path.name)
        if not m:
            raise ValueError(f"Not a Cityscapes image path: {self.image_path}")

        self.city = m["city"]
        self.clip_idx = m["clip_idx"]
        self.frame_idx = m["frame_idx"]


def _index_split_dir(
    split_dir: Path, to_datapoint: Callable[[Path], CityscapesDatapoint | None]
) -> list[CityscapesDatapoint]:
    """
    Return datapoints for `split_dir`.
    """

    if not split_dir.is_dir():
        raise FileNotFoundError(f"No split directory: '{split_dir}'")

    start = time.perf_counter()

    datapoints = []
    for path in sorted(split_dir.rglob("*")):
        if path.is_file() and (dp := to_datapoint(path)) is not None:
            datapoints.append(dp)

    logger.info(
        "Took %.2fs to index datapoints in '%s'",
        time.perf_counter() - start,
        split_dir,
    )
    return datapoints


class CityscapesUnlabeledDataset(SplitDataset):
    """
    Class to support accessing Cityscapes images, ignoring any labels.
    """

    def __init__(
        self,
        image_dir: str | Path = CITYSCAPES_DIR / "leftImg8bit_sequence",
        selected_is_oos_splits: Sequence[str] = ("train",),
        keep_every_nth_frame: int = 2,
        seed: int = DEFAULT_SEED,
    ):
        super().__init__(selected_keys=selected_is_oos_splits, seed=seed)
        if keep_every_nth_frame < 1:
            raise ValueError(f"{keep_every_nth_frame=} must be at least 1.")

        self.image_dir = Path(image_dir)
        self.keep_every_nth_frame = keep_every_nth_frame

    def __repr__(self) -> str:
        return (
            f"{super().__repr__()[:-1]}, "
            f"keep_every_nth_frame={self.keep_every_nth_frame})"
        )

    def index_key(self, is_oos_split: str) -> list[CityscapesDatapoint]:
        def to_datapoint(path: Path) -> CityscapesDatapoint | None:
            if not _IMAGE_NAME_PATTERN.match(path.name):
                return None
            return CityscapesDatapoint(image_path=path)

        datapoints = _index_split_dir(
            self.image_dir / is_oos_split, to_datapoint
        )

        clip_to_datapoints = collections.defaultdict(list)
        for dp in datapoints:
            clip_to_datapoints[dp.city, dp.clip_idx].append(dp)

        return [
            dp
            for clip_datapoints in clip_to_datapoints.values()
            for dp in clip_datapoints[:: self.keep_every_nth_frame]
        ]


_CITYSCAPES_EVAL_RAW_IDS = sorted(
    class_id
    for class_id, name in CITYSCAPES_CLASS_NAMES.items()
    if name in CITYSCAPES_EVAL_CLASSES
)

CITYSCAPES_LABEL_METADATA = LabelMetadata(
    raw_to_label={
        raw_id: train_id
        for train_id, raw_id in enumerate(_CITYSCAPES_EVAL_RAW_IDS)
    },
    eval_class_ids=frozenset(range(len(_CITYSCAPES_EVAL_RAW_IDS))),
    class_names={
        train_id: CITYSCAPES_CLASS_NAMES[raw_id]
        for train_id, raw_id in enumerate(_CITYSCAPES_EVAL_RAW_IDS)
    },
    class_colors={
        **{
            train_id: CITYSCAPES_CLASS_COLORS[raw_id]
            for train_id, raw_id in enumerate(_CITYSCAPES_EVAL_RAW_IDS)
        },
        IGNORE_LABEL_ID: IGNORE_COLOR,
    },
)


class CityscapesLabeledDataset(SplitDataset):
    """
    Class to support accessing the labeled Cityscapes dataset.
    """

    LABEL_METADATA = CITYSCAPES_LABEL_METADATA

    def __init__(
        self,
        image_dir: str | Path = CITYSCAPES_DIR / "leftImg8bit",
        label_dir: str | Path = CITYSCAPES_DIR / "gtFine",
        selected_is_oos_splits: Sequence[str] = ("train",),
        seed: int = DEFAULT_SEED,
    ):
        super().__init__(selected_keys=selected_is_oos_splits, seed=seed)
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

    def index_key(self, is_oos_split: str) -> list[CityscapesDatapoint]:
        def to_datapoint(path: Path) -> CityscapesDatapoint | None:
            m = _LABEL_NAME_PATTERN.match(path.name)
            if not m:
                return None

            image_path = (
                self.image_dir
                / is_oos_split
                / m["city"]
                / f"{m['city']}_{m['clip_idx']}_{m['frame_idx']}_leftImg8bit.png"
            )
            return CityscapesDatapoint(image_path=image_path, label_path=path)

        return _index_split_dir(self.label_dir / is_oos_split, to_datapoint)


_PERSON_NEGATIVE_ID = 0
_PERSON_POSITIVE_ID = 1

CITYSCAPES_PERSON_LABEL_METADATA = LabelMetadata(
    raw_to_label={
        class_id: (
            _PERSON_POSITIVE_ID
            if name in CITYSCAPES_PERSON_CLASSES
            else IGNORE_LABEL_ID
            if name in CITYSCAPES_VOID_CLASSES
            else _PERSON_NEGATIVE_ID
        )
        for class_id, name in CITYSCAPES_CLASS_NAMES.items()
    },
    class_names={
        _PERSON_NEGATIVE_ID: "background",
        _PERSON_POSITIVE_ID: "person",
    },
    eval_class_ids=frozenset({_PERSON_NEGATIVE_ID, _PERSON_POSITIVE_ID}),
    class_colors={
        _PERSON_NEGATIVE_ID: (0, 0, 0),
        _PERSON_POSITIVE_ID: CITYSCAPES_CLASS_COLORS[24],
        IGNORE_LABEL_ID: IGNORE_COLOR,
    },
)


class CityscapesPersonLabeledDataset(CityscapesLabeledDataset):
    LABEL_METADATA = CITYSCAPES_PERSON_LABEL_METADATA
