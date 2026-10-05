"""
Module the defines fundamental dataset structure.
"""

import functools
import itertools
import logging
from collections.abc import Iterator, Sequence

import numpy as np

from semi_sup_seg.constants import DEFAULT_SEED
from semi_sup_seg.data.datapoints import (
    DEFAULT_WORKER_COUNT,
    Datapoint,
    LoadedDatapoint,
    load_in_parallel,
)
from semi_sup_seg.data.images import get_image_size
from semi_sup_seg.data.iteration import iter_shuffled

logger = logging.getLogger(__name__)


class Dataset:
    """
    A collection of datapoints, which can be iterated over.

    Subclasses override `get_datapoints`.
    """

    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = seed

    def get_datapoints(self) -> list[Datapoint]:
        """
        Get datapoints, in stable order.
        """

        raise NotImplementedError

    def get_datapoint_count(self) -> int:
        return len(self.get_datapoints())

    def get_image_size(self) -> tuple[int, int]:
        return get_image_size(self.get_datapoints()[0].image_path)

    def iter_datapoints(
        self, num_datapoints: int | None = None, shuffle: bool = False
    ) -> Iterator[Datapoint]:
        """
        Yield `num_datapoints` datapoints, without loading them.
        """

        datapoints = self.get_datapoints()
        if num_datapoints is None:
            num_datapoints = len(datapoints)

        if shuffle:
            rng = np.random.default_rng(self.seed)
            yield from iter_shuffled(datapoints, num_datapoints, rng)
            return

        if not datapoints and num_datapoints > 0:
            raise ValueError(
                f"Can't yield {num_datapoints:,} of no datapoints."
            )
        yield from itertools.islice(itertools.cycle(datapoints), num_datapoints)

    def iter_loaded_datapoints(
        self,
        num_datapoints: int | None = None,
        shuffle: bool = False,
        worker_count: int = DEFAULT_WORKER_COUNT,
    ) -> Iterator[LoadedDatapoint]:
        """
        Yield datapoints as `iter_datapoints` does, but loaded.
        """

        logger.info(f"{self} has {self.get_datapoint_count():,} datapoints.")
        yield from load_in_parallel(
            self.iter_datapoints(num_datapoints, shuffle), worker_count
        )


DatapointIndex = dict[str, list[Datapoint]]


class SplitDataset(Dataset):
    """
    A Dataset indexed into sub-collections of datapoints.

    Subclasses override `index_key`.
    """

    def __init__(self, selected_keys: Sequence[str], seed: int = DEFAULT_SEED):
        super().__init__(seed=seed)
        if isinstance(selected_keys, str):
            raise TypeError(f"{selected_keys=} must be a sequence of keys.")
        self.selected_keys = tuple(selected_keys)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(selected_keys={list(self.selected_keys)})"
        )

    def index_key(self, key: str) -> list[Datapoint]:
        """
        Get the datapoints under a key, in a stable order.
        """

        raise NotImplementedError

    @functools.cached_property
    def datapoint_index(self) -> DatapointIndex:
        """
        Index of the selected keys.
        """

        return {key: self.index_key(key) for key in self.selected_keys}

    def get_datapoints(self) -> list[Datapoint]:
        return [
            dp
            for key_datapoints in self.datapoint_index.values()
            for dp in key_datapoints
        ]


class MixtureDataset(Dataset):
    """
    A mixture of Dataset objects, defined by `self.dataset_to_weight`.
    """

    def __init__(
        self,
        dataset_to_weight: Sequence[tuple[Dataset, float]],
        seed: int = DEFAULT_SEED,
    ):
        super().__init__(seed=seed)
        if not dataset_to_weight:
            raise ValueError("A mixture needs at least one dataset.")
        if min(weight for _, weight in dataset_to_weight) <= 0:
            raise ValueError(f"{dataset_to_weight=} weights must be positive.")
        self.dataset_to_weight = list(dataset_to_weight)

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{dataset!r}: {weight}"
            for dataset, weight in self.dataset_to_weight
        )
        return f"{type(self).__name__}({parts})"

    def get_datapoints(self) -> list[Datapoint]:
        return [
            dp
            for dataset, _ in self.dataset_to_weight
            for dp in dataset.get_datapoints()
        ]

    def get_image_size(self) -> tuple[int, int]:
        sizes = {
            dataset.get_image_size() for dataset, _ in self.dataset_to_weight
        }
        if len(sizes) != 1:
            raise ValueError(f"{self} mixes image sizes {sizes}.")
        return sizes.pop()

    def iter_datapoints(
        self, num_datapoints: int | None = None, shuffle: bool = False
    ) -> Iterator[Datapoint]:
        """
        Yield `num_datapoints` datapoints, each from a dataset chosen by weight.
        """

        if not shuffle:
            raise ValueError(f"{self} can only yield datapoints shuffled.")
        if num_datapoints is None:
            num_datapoints = self.get_datapoint_count()

        datasets = [dataset for dataset, _ in self.dataset_to_weight]
        weights = np.array([weight for _, weight in self.dataset_to_weight])
        dataset_datapoint_iters = [
            dataset.iter_datapoints(num_datapoints, shuffle=True)
            for dataset in datasets
        ]

        # A stream distinct from `default_rng(seed)`, which datasets use.
        rng = np.random.default_rng(
            np.random.SeedSequence(self.seed).spawn(1)[0]
        )
        dataset_idxs = rng.choice(
            len(datasets), size=num_datapoints, p=weights / weights.sum()
        )
        for dataset_idx in dataset_idxs:
            yield next(dataset_datapoint_iters[dataset_idx])
