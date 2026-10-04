"""
Definitions of one-off pipelines.

Example usage:
python pipelines.py
"""

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from semi_sup_seg.constants import CITYSCAPES_DIR
from semi_sup_seg.data import (
    DEFAULT_WORKER_COUNT,
    CityscapesLabeledDataset,
    CityscapesUnlabeledDataset,
    get_cityscapes_datapoint_index_iter,
)

logger = logging.getLogger(__name__)

DEFAULT_LOG_EVERY_N = 500


def _write_png_atomically(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(f"{path.name}.tmp")
    Image.fromarray(array).save(temporary_path, format="PNG")
    temporary_path.replace(path)


def downsample_image(
    input_image_path: str | Path,
    output_image_path: str | Path,
    downsample_factor: int,
) -> None:
    """
    Write a (H, W, 3) uint8 image, downsampled to (H / factor, W / factor).

    Done in a way that aligns with `downsample_label`.
    """

    f = downsample_factor

    image = np.array(Image.open(input_image_path).convert("RGB"))

    # Triangle weights at offsets -(f-1)..(f-1), e.g. [1, 2, 1] / 4 for f = 2.
    tent = 1 - torch.arange(-(f - 1), f).abs() / f
    tent = tent / tent.sum()
    kernel = (tent[:, None] * tent[None, :])[None, None]

    # (H, W, 3) --> (3, 1, H, W), so each channel is filtered on its own.
    x = torch.from_numpy(image).permute(2, 0, 1)[:, None].float()
    x = F.pad(x, (f - 1,) * 4, mode="replicate")
    x = F.conv2d(x, kernel, stride=f)  # (3, 1, ceil(H / f), ceil(W / f)).

    downsampled = x[:, 0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)

    _write_png_atomically(downsampled.numpy(), Path(output_image_path))


def downsample_label(
    input_label_path: str | Path,
    output_label_path: str | Path,
    downsample_factor: int,
) -> None:
    """
    Write a (H, W) uint8 label, downsampled to (H / factor, W / factor).

    Done in a way that aligns with `downsample_image`.
    """

    label = np.asarray(Image.open(input_label_path))
    downsampled = np.ascontiguousarray(
        label[::downsample_factor, ::downsample_factor]
    )

    _write_png_atomically(downsampled, Path(output_label_path))


# (downsampling function, input path, output path), per file to write.
_Job = tuple[Callable[[Path, Path, int], None], Path, Path]


def _write_files(
    jobs: list[_Job],
    downsample_factor: int,
    datapoint_count: int,
    force: bool,
    worker_count: int,
) -> None:
    """
    Run `jobs` with `worker_count` threads, skipping existing outputs unless
    `force`.
    """

    skipped_count = 0
    if not force:
        unskipped_jobs = [job for job in jobs if not job[2].exists()]
        skipped_count = len(jobs) - len(unskipped_jobs)
        jobs = unskipped_jobs

    logger.info(
        f"Writing {len(jobs):,} files, {downsample_factor}x downsampled, from "
        f"{datapoint_count:,} datapoints. Skipping {skipped_count:,} that "
        "exist."
    )

    start = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="downsample"
    ) as pool:
        futures = [
            pool.submit(downsample, input_path, output_path, downsample_factor)
            for downsample, input_path, output_path in jobs
        ]
        for written_count, future in enumerate(futures, start=1):
            future.result()

            is_last = written_count == len(futures)
            if written_count % DEFAULT_LOG_EVERY_N == 0 or is_last:
                logger.info(
                    f"Wrote {written_count:,}/{len(futures):,} files in "
                    f"{time.perf_counter() - start:.0f}s."
                )


def write_downsampled_cityscapes_labeled_dataset(
    input_image_dir: str | Path,
    input_label_dir: str | Path,
    output_image_dir: str | Path,
    output_label_dir: str | Path,
    downsample_factor: int,
    force: bool = False,  # Overwrite existing files, instead of skipping them.
    worker_count: int = DEFAULT_WORKER_COUNT,
) -> None:
    """
    Write downsampled copies of every finely labeled (image, label) pair.

    Outputs mirror the inputs' directory layout & file names, so
    `CityscapesLabeledDataset(output_image_dir, output_label_dir)` reads them.
    """

    input_image_dir = Path(input_image_dir)
    input_label_dir = Path(input_label_dir)
    output_image_dir = Path(output_image_dir)
    output_label_dir = Path(output_label_dir)

    labeled_dataset = CityscapesLabeledDataset(
        image_dir=input_image_dir, fine_label_dir=input_label_dir
    )
    datapoints = list(
        get_cityscapes_datapoint_index_iter(
            labeled_dataset.fine_datapoint_index
        )
    )

    jobs: list[_Job] = []
    for dp in datapoints:
        jobs.append(
            (
                downsample_image,
                dp.image.path,
                output_image_dir / dp.image.path.relative_to(input_image_dir),
            )
        )
        jobs.append(
            (
                downsample_label,
                dp.label.path,
                output_label_dir / dp.label.path.relative_to(input_label_dir),
            )
        )

    _write_files(jobs, downsample_factor, len(datapoints), force, worker_count)


def write_downsampled_cityscapes_unlabeled_dataset(
    input_image_dir: str | Path,
    output_image_dir: str | Path,
    downsample_factor: int,
    splits: tuple[str, ...] = ("train",),  # IS-OOS splits to write.
    force: bool = False,  # Overwrite existing files, instead of skipping them.
    worker_count: int = DEFAULT_WORKER_COUNT,
) -> None:
    """
    Write downsampled copies of every image in `splits`.

    Outputs mirror the inputs' directory layout & file names, so
    `CityscapesUnlabeledDataset(output_image_dir)` reads them.
    """

    input_image_dir = Path(input_image_dir)
    output_image_dir = Path(output_image_dir)

    unlabeled_dataset = CityscapesUnlabeledDataset(image_dir=input_image_dir)
    datapoints = list(
        get_cityscapes_datapoint_index_iter(
            unlabeled_dataset.image_index, selected_is_oos_splits=list(splits)
        )
    )

    jobs: list[_Job] = [
        (
            downsample_image,
            dp.image.path,
            output_image_dir / dp.image.path.relative_to(input_image_dir),
        )
        for dp in datapoints
    ]

    _write_files(jobs, downsample_factor, len(datapoints), force, worker_count)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    torch.set_num_threads(1)  # Parallelism comes from worker threads.

    write_downsampled_cityscapes_labeled_dataset(
        input_image_dir=CITYSCAPES_DIR / "leftImg8bit",
        input_label_dir=CITYSCAPES_DIR / "gtFine",
        output_image_dir=CITYSCAPES_DIR / "leftImg8bit_2x_downsampled",
        output_label_dir=CITYSCAPES_DIR / "gtFine_2x_downsampled",
        downsample_factor=2,
        force=True,
    )
    # write_downsampled_cityscapes_unlabeled_dataset(
    #     input_image_dir=CITYSCAPES_DIR / "leftImg8bit_sequence",
    #     output_image_dir=CITYSCAPES_DIR / "leftImg8bit_sequence_2x_downsampled",
    #     downsample_factor=2,
    # )
