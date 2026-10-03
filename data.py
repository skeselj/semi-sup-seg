"""
Module to support accessing semantic segmentation data.
"""

import collections
import dataclasses
import itertools
import logging
import re
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Self

import numpy as np
from PIL import Image

from constants import (
    CITYSCAPES_CLASS_COLORS,
    CITYSCAPES_CLASS_NAMES,
    CITYSCAPES_DIR,
    CITYSCAPES_EVAL_CLASSES,
    CITYSCAPES_PERSON_CLASSES,
    CITYSCAPES_VOID_CLASSES,
    DEFAULT_SEED,
    IGNORE_LABEL_ID,
    MAX_LABEL_COUNT,
)

logger = logging.getLogger(__name__)

DEFAULT_WORKER_COUNT = 12


class InvalidPathException(Exception):
    pass


@dataclasses.dataclass(frozen=True, kw_only=True)
class LabelMetadata:
    """
    Metadata that informs how to work with a given set of integer labels.
    """

    class_names: dict[int, str]  # Label value --> class name.
    ignore_id: int = IGNORE_LABEL_ID  # Label value of ignored pixels.
    eval_class_ids: frozenset[int]  # Classes trained & evaluated on.
    class_colors: dict[int, tuple[int, int, int]]  # Label value --> RGB color.


@dataclasses.dataclass
class _CityscapesFile:
    """
    A Cityscapes file, identified by its name: <city>_<clip>_<frame>_<suffix>.
    """

    FILE_NAME_PATTERN = re.compile(r"(?!)")  # Matches nothing.

    city: str
    clip_idx: str
    frame_idx: str

    path: Path
    ary: np.ndarray | None = dataclasses.field(default=None, repr=False)

    def __init__(self, path: str | Path):
        self.path = Path(path)
        m = self.FILE_NAME_PATTERN.match(self.path.name)
        if not m:
            raise InvalidPathException(
                f"Not a {type(self).__name__} path: {path}"
            )

        self.ary = None

        self.city = m["city"]
        self.clip_idx = m["clip_idx"]
        self.frame_idx = m["frame_idx"]

    def load(self) -> None:
        raise NotImplementedError


class _CityscapesImage(_CityscapesFile):
    FILE_NAME_PATTERN = re.compile(
        r"^(?P<city>[a-z-]+)_(?P<clip_idx>\d{6})_(?P<frame_idx>\d{6})"
        r"_leftImg8bit\.png$"
    )

    @classmethod
    def get_path_name(cls, city: str, clip_idx: str, frame_idx: str) -> str:
        return f"{city}_{clip_idx}_{frame_idx}_leftImg8bit.png"

    def load(self) -> None:
        """
        Load the image into self.ary as (H, W, 3) uint8 RGB intensities.
        """
        self.ary = np.asarray(Image.open(self.path).convert("RGB"))


class _CityscapesLabel(_CityscapesFile):
    FILE_NAME_PATTERN = re.compile(
        r"^(?P<city>[a-z-]+)_(?P<clip_idx>\d{6})_(?P<frame_idx>\d{6})"
        r"_gt(?:Fine|Coarse)_labelIds\.png$"
    )

    @classmethod
    def get_path_name(
        cls,
        city: str,
        clip_idx: str,
        frame_idx: str,
        granularity: Literal["Fine", "Coarse"],
    ) -> str:
        return f"{city}_{clip_idx}_{frame_idx}_gt{granularity}_labelIds.png"

    def load(self) -> None:
        """
        Load the label into self.ary as (H, W) uint8 class IDs.
        """
        self.ary = np.asarray(Image.open(self.path))


@dataclasses.dataclass
class CityscapesDatapoint:
    image: _CityscapesImage
    label: _CityscapesLabel

    def loaded(self) -> Self:
        """
        Return a copy of this datapoint, with its image & label loaded.
        """

        dp = CityscapesDatapoint(
            image=_CityscapesImage(self.image.path),
            label=_CityscapesLabel(self.label.path),
        )
        dp.image.load()
        dp.label.load()
        return dp


_CityscapesDatapointIndex = dict[
    # IS-OOS split, (e.g. "train_extra") -->
    str,
    dict[
        # city (e.g. "cologne") -->
        str,
        dict[
            # clip index (e.g. "000050") --> datapoints
            str, list[CityscapesDatapoint]
        ],
    ],
]


def _get_cityscapes_datapoint_index_iter(
    datapoint_index: _CityscapesDatapointIndex,
    selected_is_oos_splits: list[str] | None = None,
) -> Iterator[CityscapesDatapoint]:
    """
    Get iterator over datapoints in datapoint index.
    """

    if selected_is_oos_splits is None:
        selected_is_oos_splits = datapoint_index.keys()

    for is_oos_split in selected_is_oos_splits:
        split_index = datapoint_index.get(is_oos_split, {})
        for city in split_index:
            for clip_idx in split_index[city]:
                yield from split_index[city][clip_idx]


def _load_in_parallel(
    datapoints: Iterable[CityscapesDatapoint],
    worker_count: int = DEFAULT_WORKER_COUNT,
) -> Iterator[CityscapesDatapoint]:
    """
    Yield loaded copies of `datapoints`, in order, with worker count threads.
    """

    if worker_count < 1:
        raise ValueError(f"{worker_count=} must be at least 1.")

    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="data-load"
    ) as pool:
        pending: collections.deque[Future[CityscapesDatapoint]] = (
            collections.deque()
        )
        try:
            for dp in datapoints:
                pending.append(pool.submit(dp.loaded))
                if len(pending) >= 2 * worker_count:
                    yield pending.popleft().result()

            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()


class CityscapesLabeledDataset:
    """
    Class to support accessing the labeled Cityscapes dataset.
    """

    IS_OOS_SPLITS = ("train", "train_extra", "val", "test")

    LABEL_METADATA = LabelMetadata(
        class_names=CITYSCAPES_CLASS_NAMES,
        eval_class_ids=frozenset(
            class_id
            for class_id, name in CITYSCAPES_CLASS_NAMES.items()
            if name in CITYSCAPES_EVAL_CLASSES
        ),
        class_colors=CITYSCAPES_CLASS_COLORS,
    )

    def _construct_datapoint_index(
        self, label_dir: str | Path
    ) -> _CityscapesDatapointIndex:
        label_dir = Path(label_dir)

        ret = (
            # IS-OOS split --> ...
            collections.defaultdict(
                # city --> ...
                lambda: collections.defaultdict(
                    # clip index --> datapoints
                    lambda: collections.defaultdict(list)
                )
            )
        )

        for label_path in sorted(label_dir.rglob("*")):
            if not label_path.is_file():
                continue

            try:
                label = _CityscapesLabel(label_path)
            except InvalidPathException:
                continue

            is_oos_split = label_path.relative_to(label_dir).parts[0]
            image = _CityscapesImage(
                self.image_dir
                / is_oos_split
                / label.city
                / _CityscapesImage.get_path_name(
                    city=label.city,
                    clip_idx=label.clip_idx,
                    frame_idx=label.frame_idx,
                )
            )

            ret[is_oos_split][label.city][label.clip_idx].append(
                CityscapesDatapoint(image=image, label=label)
            )

        return ret

    def __init__(
        self,
        image_dir: str | Path = CITYSCAPES_DIR / "leftImg8bit",
        fine_label_dir: str | Path = CITYSCAPES_DIR / "gtFine",
        coarse_label_dir: str | Path = CITYSCAPES_DIR / "gtCoarse",
        seed: int = DEFAULT_SEED,
    ):
        self.image_dir = Path(image_dir)
        self.fine_label_dir = Path(fine_label_dir)
        self.coarse_label_dir = Path(coarse_label_dir)

        self.seed = seed

        start = time.perf_counter()
        self.fine_datapoint_index = self._construct_datapoint_index(
            self.fine_label_dir
        )
        self.coarse_datapoint_index = self._construct_datapoint_index(
            self.coarse_label_dir
        )
        logger.info(
            "Took %.2fs to index fine and coarse datapoints",
            time.perf_counter() - start,
        )

    def iter_fine_train_datapoints(
        self,
        num_datapoints: int | None = None,
        worker_count: int = DEFAULT_WORKER_COUNT,
    ) -> Iterator[CityscapesDatapoint]:
        """
        Return an iterator over train datapoints, in random order.
        """

        rng = np.random.default_rng(self.seed)

        datapoints = list(
            _get_cityscapes_datapoint_index_iter(
                self.fine_datapoint_index,
                selected_is_oos_splits=["train", "train_extra"],
            )
        )
        logger.info(
            f"There are {len(datapoints):} unique finely-labeled Cityscapes "
            "train datapoints"
        )

        if num_datapoints is None:
            num_datapoints = len(datapoints)

        def shuffled_idxs_gen() -> Iterator[int]:
            while True:
                yield from rng.permutation(len(datapoints))

        yield from _load_in_parallel(
            (
                datapoints[i]
                for i in itertools.islice(shuffled_idxs_gen(), num_datapoints)
            ),
            worker_count,
        )

    def _get_fine_val_datapoints(self) -> list[CityscapesDatapoint]:
        return list(
            _get_cityscapes_datapoint_index_iter(
                self.fine_datapoint_index,
                selected_is_oos_splits=["val"],
            )
        )

    def get_fine_val_datapoint_count(self) -> int:
        return len(self._get_fine_val_datapoints())

    def iter_fine_val_datapoints(
        self,
        shuffle: bool = False,
        worker_count: int = DEFAULT_WORKER_COUNT,
    ) -> Iterator[CityscapesDatapoint]:
        """
        Return an iterator over val datapoints, in index order.
        """

        datapoints = self._get_fine_val_datapoints()
        logger.info(
            f"There are {len(datapoints):} finely-labeled Cityscapes val "
            "datapoints"
        )

        if shuffle:
            rng = np.random.default_rng(self.seed)
            datapoints = [
                datapoints[i] for i in rng.permutation(len(datapoints))
            ]

        yield from _load_in_parallel(datapoints, worker_count)


class CityscapesPersonLabeledDataset(CityscapesLabeledDataset):
    """
    Class to support accessing the "person" labeled Cityscapes dataset.
    """

    NEGATIVE_ID = 0
    POSITIVE_ID = 1
    IGNORE_ID = IGNORE_LABEL_ID

    LABEL_METADATA = LabelMetadata(
        class_names={NEGATIVE_ID: "background", POSITIVE_ID: "person"},
        ignore_id=IGNORE_ID,
        eval_class_ids=frozenset({NEGATIVE_ID, POSITIVE_ID}),
        class_colors={
            NEGATIVE_ID: (0, 0, 0),
            POSITIVE_ID: CITYSCAPES_CLASS_COLORS[24],
            IGNORE_ID: (128, 128, 128),
        },
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # General Cityscapes class ID --> this dataset class ID.
        self.label_map = np.full(
            MAX_LABEL_COUNT, self.NEGATIVE_ID, dtype=np.uint8
        )
        for class_id, name in CITYSCAPES_CLASS_NAMES.items():
            if name in CITYSCAPES_PERSON_CLASSES:
                self.label_map[class_id] = self.POSITIVE_ID
            elif name in CITYSCAPES_VOID_CLASSES:
                self.label_map[class_id] = self.IGNORE_ID

    def _to_these_labels(
        self, datapoints: Iterator[CityscapesDatapoint]
    ) -> Iterator[CityscapesDatapoint]:
        """
        Map the labels of loaded `datapoints` to this dataset's labels.
        """

        for dp in datapoints:
            dp.label.ary = self.label_map[dp.label.ary]
            yield dp

    def iter_fine_train_datapoints(
        self,
        num_datapoints: int | None = None,
        worker_count: int = DEFAULT_WORKER_COUNT,
    ) -> Iterator[CityscapesDatapoint]:
        return self._to_these_labels(
            super().iter_fine_train_datapoints(num_datapoints, worker_count)
        )

    def iter_fine_val_datapoints(
        self,
        shuffle: bool = False,
        worker_count: int = DEFAULT_WORKER_COUNT,
    ) -> Iterator[CityscapesDatapoint]:
        return self._to_these_labels(
            super().iter_fine_val_datapoints(shuffle, worker_count)
        )


# Dataset class name --> class.
CITYSCAPES_DATASET_CLASSES: dict[str, type[CityscapesLabeledDataset]] = {
    cls.__name__: cls
    for cls in (CityscapesLabeledDataset, CityscapesPersonLabeledDataset)
}
