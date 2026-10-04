"""
Module to support accessing semantic segmentation data.
"""

import collections
import dataclasses
import functools
import logging
import re
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from PIL import Image

from constants import CITYSCAPES_DIR, DEFAULT_SEED
from data._iter import TensorDatapointIter as TensorDatapointIter
from data._iter import batch as batch
from data._iter import iter_shuffled
from data._iter import prefetch as prefetch
from data.labels import (
    CITYSCAPES_LABEL_METADATA,
    CITYSCAPES_PERSON_LABEL_METADATA,
)

logger = logging.getLogger(__name__)

DEFAULT_WORKER_COUNT = 12


class InvalidPathException(Exception):
    pass


@dataclasses.dataclass
class _CityscapesFile:
    """
    A Cityscapes file, identified by its name: <city>_<clip>_<frame>_<suffix>.
    """

    FILE_NAME_PATTERN = re.compile(r"(?!)")  # Matches nothing.

    path: Path
    city: str = dataclasses.field(init=False)
    clip_idx: str = dataclasses.field(init=False)
    frame_idx: str = dataclasses.field(init=False)

    def __post_init__(self):
        self.path = Path(self.path)
        m = self.FILE_NAME_PATTERN.match(self.path.name)
        if not m:
            raise InvalidPathException(
                f"Not a {type(self).__name__} path: {self.path}"
            )

        self.city = m["city"]
        self.clip_idx = m["clip_idx"]
        self.frame_idx = m["frame_idx"]

    def load(self) -> np.ndarray:
        raise NotImplementedError


class _CityscapesImage(_CityscapesFile):
    """
    A Cityscapes file containing a (H, W, 3) uint8 image.
    """

    FILE_NAME_PATTERN = re.compile(
        r"^(?P<city>[a-z-]+)_(?P<clip_idx>\d{6})_(?P<frame_idx>\d{6})"
        r"_leftImg8bit\.png$"
    )

    @classmethod
    def get_path_name(cls, city: str, clip_idx: str, frame_idx: str) -> str:
        return f"{city}_{clip_idx}_{frame_idx}_leftImg8bit.png"

    def load(self) -> np.ndarray:
        return np.asarray(Image.open(self.path).convert("RGB"))


class _CityscapesLabel(_CityscapesFile):
    """
    A Cityscapes file containing a (H, W) uint8 label.
    """

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

    def load(self) -> np.ndarray:
        return np.asarray(Image.open(self.path))


@dataclasses.dataclass
class CityscapesUnlabeledDatapoint:
    image: _CityscapesImage

    def load(self) -> tuple[np.ndarray]:
        """
        Load (image,).
        """

        return (self.image.load(),)


@dataclasses.dataclass
class CityscapesLabeledDatapoint:
    image: _CityscapesImage
    label: _CityscapesLabel

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load (image, label).
        """

        return self.image.load(), self.label.load()


_DatapointT = TypeVar(
    "_DatapointT", CityscapesUnlabeledDatapoint, CityscapesLabeledDatapoint
)

_CityscapesDatapointIndex = dict[
    # IS-OOS split, (e.g. "train_extra") -->
    str,
    dict[
        # city (e.g. "cologne") -->
        str,
        dict[
            # clip index (e.g. "000050") --> datapoints
            str, list[_DatapointT]
        ],
    ],
]
_CityscapesUnlabeledDatapointIndex = _CityscapesDatapointIndex[
    CityscapesUnlabeledDatapoint
]
_CityscapesLabeledDatapointIndex = _CityscapesDatapointIndex[
    CityscapesLabeledDatapoint
]


def _get_empty_index() -> _CityscapesDatapointIndex:
    return (
        # IS-OOS split --> ...
        collections.defaultdict(
            # city --> ...
            lambda: collections.defaultdict(
                # clip index --> datapoints
                lambda: collections.defaultdict(list)
            )
        )
    )


def _build_index(
    root_dir: Path, to_datapoint: Callable[[Path, str], _DatapointT]
) -> _CityscapesDatapointIndex[_DatapointT]:
    """
    Index datapoints, one per file under `root_dir`.
    """

    start = time.perf_counter()

    ret = _get_empty_index()
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue

        is_oos_split = path.relative_to(root_dir).parts[0]
        try:
            dp = to_datapoint(path, is_oos_split)
        except InvalidPathException:
            continue

        ret[is_oos_split][dp.image.city][dp.image.clip_idx].append(dp)

    logger.info(
        "Took %.2fs to index datapoints in '%s'",
        time.perf_counter() - start,
        root_dir,
    )
    return ret


def _get_cityscapes_datapoint_index_iter(
    datapoint_index: _CityscapesDatapointIndex[_DatapointT],
    selected_is_oos_splits: list[str] | None = None,
) -> Iterator[_DatapointT]:
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
    datapoints: Iterable[_DatapointT],
    worker_count: int = DEFAULT_WORKER_COUNT,
) -> Iterator[tuple[np.ndarray, ...]]:
    """
    Yield loaded `datapoints`, in order, with worker count threads.
    """

    if worker_count < 1:
        raise ValueError(f"{worker_count=} must be at least 1.")

    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="data-load"
    ) as pool:
        pending: collections.deque[Future[tuple[np.ndarray, ...]]] = (
            collections.deque()
        )
        try:
            for dp in datapoints:
                pending.append(pool.submit(dp.load))
                if len(pending) >= 2 * worker_count:
                    yield pending.popleft().result()

            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()


def _iter_loaded(
    datapoints: list[_DatapointT],
    description: str,
    worker_count: int,
    rng: np.random.Generator | None = None,
    num_datapoints: int | None = None,
) -> Iterator[tuple[np.ndarray, ...]]:
    """
    Log how many `datapoints` there are, and yield them loaded.
    """

    logger.info(f"There are {len(datapoints):,} {description}.")

    selected: Iterable[_DatapointT] = datapoints
    if rng is not None:
        if num_datapoints is None:
            num_datapoints = len(datapoints)
        selected = iter_shuffled(datapoints, rng, num_datapoints)

    yield from _load_in_parallel(selected, worker_count)


class CityscapesUnlabeledDataset:
    """
    Class to support accessing Cityscapes images, ignoring any labels.
    """

    def __init__(
        self,
        image_dir: str | Path = CITYSCAPES_DIR / "leftImg8bit_sequence",
        seed: int = DEFAULT_SEED,
    ):
        self.image_dir = Path(image_dir)
        self.seed = seed

    @functools.cached_property
    def image_index(self) -> _CityscapesUnlabeledDatapointIndex:
        """
        Index of the images in `self.image_dir`, built on first access.
        """

        return _build_index(
            self.image_dir,
            lambda path, _: CityscapesUnlabeledDatapoint(
                image=_CityscapesImage(path)
            ),
        )

    def iter_train_datapoints(
        self,
        num_datapoints: int | None = None,
        worker_count: int = DEFAULT_WORKER_COUNT,
    ) -> Iterator[tuple[np.ndarray]]:
        """
        Yield loaded (image,) train datapoints, in random order.
        """

        yield from _iter_loaded(
            list(
                _get_cityscapes_datapoint_index_iter(
                    self.image_index, selected_is_oos_splits=["train"]
                )
            ),
            "unique unlabeled Cityscapes train datapoints",
            worker_count,
            rng=np.random.default_rng(self.seed),
            num_datapoints=num_datapoints,
        )


class CityscapesLabeledDataset:
    """
    Class to support accessing the labeled Cityscapes dataset.
    """

    IS_OOS_SPLITS = ("train", "train_extra", "val", "test")
    LABEL_METADATA = CITYSCAPES_LABEL_METADATA

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

    def _to_datapoint(
        self, label_path: Path, is_oos_split: str
    ) -> CityscapesLabeledDatapoint:
        """
        Get the datapoint of a label file, pairing it with its image.
        """

        label = _CityscapesLabel(label_path)
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
        return CityscapesLabeledDatapoint(image=image, label=label)

    @functools.cached_property
    def fine_datapoint_index(self) -> _CityscapesLabeledDatapointIndex:
        """
        Index of the finely labeled datapoints, built on first access.
        """

        return _build_index(self.fine_label_dir, self._to_datapoint)

    @functools.cached_property
    def coarse_datapoint_index(self) -> _CityscapesLabeledDatapointIndex:
        """
        Index of the coarsely labeled datapoints, built on first access.
        """

        return _build_index(self.coarse_label_dir, self._to_datapoint)

    def iter_fine_train_datapoints(
        self,
        num_datapoints: int | None = None,
        worker_count: int = DEFAULT_WORKER_COUNT,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yield loaded (image, label) train datapoints, in random order.
        """

        yield from _iter_loaded(
            list(
                _get_cityscapes_datapoint_index_iter(
                    self.fine_datapoint_index,
                    selected_is_oos_splits=["train"],
                )
            ),
            "unique finely-labeled Cityscapes train datapoints",
            worker_count,
            rng=np.random.default_rng(self.seed),
            num_datapoints=num_datapoints,
        )

    def _get_fine_val_datapoints(self) -> list[CityscapesLabeledDatapoint]:
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
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yield loaded (image, label) val datapoints, each once.

        They're in index order, or a random one if `shuffle`.
        """

        yield from _iter_loaded(
            self._get_fine_val_datapoints(),
            "finely-labeled Cityscapes val datapoints",
            worker_count,
            rng=np.random.default_rng(self.seed) if shuffle else None,
        )


class CityscapesPersonLabeledDataset(CityscapesLabeledDataset):
    """
    Class to support accessing the "person" labeled Cityscapes dataset.
    """

    # Labels are yielded raw, as in the data files. This metadata maps them.
    LABEL_METADATA = CITYSCAPES_PERSON_LABEL_METADATA
