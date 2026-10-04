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

from data import CityscapesLabeledDatapoint, CityscapesUnlabeledDatapoint

DEFAULT_PREFETCH_DEPTH = 16

T = TypeVar("T")
TensorDatapointIter = Iterator[tuple[torch.Tensor, torch.Tensor]]


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
    data_iter: Iterator[CityscapesLabeledDatapoint], batch_size: int
) -> TensorDatapointIter:
    """
    Yield uint8 (B, H, W, 3) image and (B, H, W) label batches.

    Batches are pinned when CUDA is available, for async host-to-device copy.
    """

    pin = torch.cuda.is_available()

    while datapoints := list(itertools.islice(data_iter, batch_size)):
        image_batch = torch.from_numpy(
            np.stack([dp.image.ary for dp in datapoints])
        )
        label_batch = torch.from_numpy(
            np.stack([dp.label.ary for dp in datapoints])
        )

        if pin:
            image_batch = image_batch.pin_memory()
            label_batch = label_batch.pin_memory()

        yield image_batch, label_batch


def batch_images(
    data_iter: Iterator[CityscapesUnlabeledDatapoint], batch_size: int
) -> Iterator[torch.Tensor]:
    """
    Yield uint8 (B, H, W, 3) image batches, ignoring any labels.

    Batches are pinned when CUDA is available, for async host-to-device copy.
    """

    pin = torch.cuda.is_available()

    while datapoints := list(itertools.islice(data_iter, batch_size)):
        image_batch = torch.from_numpy(
            np.stack([dp.image.ary for dp in datapoints])
        )

        if pin:
            image_batch = image_batch.pin_memory()

        yield image_batch
