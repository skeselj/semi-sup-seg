"""
Module the defines fundamental datapoint structure.
"""

import collections
import dataclasses
import itertools
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from semi_sup_seg.data.images import load_image, load_label

DEFAULT_WORKER_COUNT = 12


@dataclasses.dataclass
class LoadedDatapoint:
    """
    A Datapoint, with image and (optionally) label content loaded.
    """

    image: np.ndarray  # (H, W, 3) uint8.
    label: np.ndarray | None = None  # (H, W) uint8.


@dataclasses.dataclass
class Datapoint:
    """
    Information that identifies an image and (optionally) a label for it.

    Can be subclassed per dataset, e.g. to add what's parsed from file names.
    """

    image_path: Path
    label_path: Path | None = None

    def load(self) -> LoadedDatapoint:
        return LoadedDatapoint(
            image=load_image(self.image_path),
            label=(
                None if self.label_path is None else load_label(self.label_path)
            ),
        )


def load_in_parallel(
    datapoints: Iterable[Datapoint],
    worker_count: int = DEFAULT_WORKER_COUNT,
) -> Iterator[LoadedDatapoint]:
    """
    Yield loaded `datapoints`, in order, with worker count threads.
    """

    if worker_count < 1:
        raise ValueError(f"{worker_count=} must be at least 1.")

    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="data-load"
    ) as pool:
        pending: collections.deque[Future[LoadedDatapoint]] = (
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


def batch_datapoints(
    data_iter: Iterator[LoadedDatapoint], batch_size: int
) -> Iterator[tuple[torch.Tensor, ...]]:
    """
    Yield batches of loaded datapoints.
    """

    pin = torch.cuda.is_available()

    while datapoints := list(itertools.islice(data_iter, batch_size)):
        arrays = [np.stack([dp.image for dp in datapoints])]
        labels = [dp.label for dp in datapoints]
        if all(label is not None for label in labels):
            arrays.append(np.stack(labels))
        elif any(label is not None for label in labels):
            raise ValueError("Can't batch labeled & unlabeled datapoints.")

        tensors = tuple(torch.from_numpy(array) for array in arrays)

        if pin:
            tensors = tuple(tensor.pin_memory() for tensor in tensors)

        yield tensors
