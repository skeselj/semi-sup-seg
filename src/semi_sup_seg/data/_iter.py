"""
Module to support iterating over data.
"""

import itertools
import queue
import threading
from collections.abc import Iterator
from typing import TypeVar

import numpy as np
import torch

DEFAULT_PREFETCH_DEPTH = 16

T = TypeVar("T")
TensorDatapointIter = Iterator[tuple[torch.Tensor, torch.Tensor]]


def iter_shuffled(
    items: list[T], rng: np.random.Generator, count: int
) -> Iterator[T]:
    """
    Yield `count` of `items`, reshuffling `items` after each pass over them.
    """

    if not items and count > 0:
        raise ValueError(f"Can't yield {count:,} items from no items.")

    def shuffled_idxs_gen() -> Iterator[int]:
        while True:
            yield from rng.permutation(len(items))

    for i in itertools.islice(shuffled_idxs_gen(), count):
        yield items[i]


def prefetch(
    data_iter: Iterator[T], depth: int = DEFAULT_PREFETCH_DEPTH
) -> Iterator[T]:
    """
    Yield from `data_iter`, loading up to `depth` items ahead.
    """

    if depth < 0:
        raise ValueError(f"{depth=} must be at least 0.")

    if depth == 0:
        yield from data_iter
        return

    loaded: queue.Queue = queue.Queue(maxsize=depth)
    stop_event = threading.Event()
    done_indicator = object()

    def load() -> None:
        try:
            for item in data_iter:
                while not stop_event.is_set():
                    try:
                        loaded.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue

                if stop_event.is_set():
                    return
        except Exception as exc:  # noqa: BLE001
            loaded.put(exc)
        finally:
            if not stop_event.is_set():
                loaded.put(done_indicator)

    thread = threading.Thread(target=load, daemon=True, name="data-prefetch")
    thread.start()

    try:
        while True:
            item = loaded.get()

            if item is done_indicator:
                return
            if isinstance(item, Exception):
                raise item

            yield item
    finally:
        stop_event.set()


def batch(
    data_iter: Iterator[tuple[np.ndarray, ...]], batch_size: int
) -> Iterator[tuple[torch.Tensor, ...]]:
    """
    Yield batches of loaded datapoints, e.g. uint8 (B, H, W, 3) image and
    (B, H, W) label batches.
    """

    pin = torch.cuda.is_available()

    while datapoints := list(itertools.islice(data_iter, batch_size)):
        tensors = tuple(
            torch.from_numpy(np.stack(arrays)) for arrays in zip(*datapoints)
        )

        if pin:
            tensors = tuple(tensor.pin_memory() for tensor in tensors)

        yield tensors
