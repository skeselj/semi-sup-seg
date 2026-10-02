"""
Module to support accessing semantic segmentation data.
"""

import collections
import dataclasses
import logging
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from constants import CITYSCAPES_DIR, DEFAULT_SEED

logger = logging.getLogger(__name__)
DEFAULT_WORKER_COUNT = 6


class InvalidPathException(Exception):
    pass


@dataclasses.dataclass
class _CityscapesFile:
    """
    A Cityscapes file identified by its name: <city>_<clip>_<frame>_<suffix>.

    Subclasses set FILE_NAME_PATTERN and implement load(). The array is only
    read from disk on load().
    """

    FILE_NAME_PATTERN = re.compile(r"(?!)")  # matches nothing; see subclasses

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
class _CityscapesDatapoint:
    image: _CityscapesImage
    label: _CityscapesLabel

    def load(self) -> None:
        self.image.load()
        self.label.load()


# IS-OOS split --> city --> clip index --> [datapoint], in sorted path order.
_DatapointIndex = dict[
    str, dict[str, dict[str, list[_CityscapesDatapoint]]]
]


class CityscapesDataset:
    """
    Class to support accessing the Cityscapes dataset.

    Levels of organization: (IS-OOS split) / (city) / (video clip) / (frame).
    """

    IS_OOS_SPLITS = ("train", "train_extra", "val", "test")

    def _construct_datapoint_index(
        self, label_dir: str | Path
    ) -> _DatapointIndex:
        label_dir = Path(label_dir)

        ret = (
            # is-oos split --> ...
            collections.defaultdict(
                # city --> ...
                lambda: collections.defaultdict(
                    # clip index --> [datapoint]
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
                Path(self.image_dir)
                / is_oos_split
                / label.city
                / _CityscapesImage.get_path_name(
                    city=label.city,
                    clip_idx=label.clip_idx,
                    frame_idx=label.frame_idx,
                )
            )

            ret[is_oos_split][label.city][label.clip_idx].append(
                _CityscapesDatapoint(image=image, label=label)
            )

        return ret

    def _get_datapoint_index_iter(
        self,
        datapoint_index: _DatapointIndex,
        selected_is_oos_splits: list[str] | None = None,
    ) -> Iterator[_CityscapesDatapoint]:
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

    def __init__(
        self,
        image_dir: str | Path = CITYSCAPES_DIR / "leftImg8bit",
        fine_label_dir: str | Path = CITYSCAPES_DIR / "gtFine",
        coarse_label_dir: str | Path = CITYSCAPES_DIR / "gtCoarse",
        seed: int = DEFAULT_SEED,
    ):
        self.image_dir = image_dir
        self.fine_label_dir = fine_label_dir
        self.coarse_label_dir = coarse_label_dir

        self.seed = seed

        start = time.perf_counter()
        self.fine_datapoint_index = self._construct_datapoint_index(
            fine_label_dir
        )
        self.coarse_datapoint_index = self._construct_datapoint_index(
            coarse_label_dir
        )
        logger.info(
            "Took %.2fs to index fine and coarse datapoints",
            time.perf_counter() - start,
        )

    def get_fine_label_train_dataset(self) -> Iterator[_CityscapesDatapoint]:
        """Return an iterator over train datapoints, in random order."""

        rng = np.random.default_rng(self.seed)

        datapoints = list(
            self._get_datapoint_index_iter(
                self.fine_datapoint_index,
                selected_is_oos_splits=["train", "train_extra"],
            )
        )
        for i in rng.permutation(len(datapoints)):
            dp = datapoints[i]
            dp.load()
            yield dp


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    cityscapes_dataset = CityscapesDataset()

    start = time.perf_counter()
    
    for idx, dp in enumerate(cityscapes_dataset.get_fine_label_train_dataset()):
        logger.info(
            f"{idx=}: \n"
            f"\t{dp.image.ary.shape=}, {dp.image.ary.min()=} {dp.image.ary.max()=}, \n"
            f"\t{dp.label.ary.shape=}, {dp.label.ary.min()=} {dp.label.ary.max()=}"
        )

        if idx > 10:
            break
    
    logger.info(
        "Took %.2fs to check on the dataset",
        time.perf_counter() - start,
    )
    