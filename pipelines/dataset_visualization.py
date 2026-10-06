"""
Pipeline to write videos of datasets, for viewing.

Example usage:
python pipelines/dataset_visualization.py
"""

import collections
import functools
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from semi_sup_seg.constants import (
    CITYSCAPES_CLASS_COLORS,
    CITYSCAPES_DIR,
    MAX_LABEL_COUNT,
)
from semi_sup_seg.data.cityscapes import (
    CityscapesDatapoint,
    CityscapesLabeledDataset,
    CityscapesUnlabeledDataset,
)
from semi_sup_seg.data.datapoints import DEFAULT_WORKER_COUNT, load_in_parallel
from semi_sup_seg.data.labels import IGNORE_COLOR

logger = logging.getLogger(__name__)

DEFAULT_FPS = 17  # Cityscapes sequences are recorded at 17 Hz.
DEFAULT_LABELED_FRAME_HOLD_SECONDS = 1.0
DEFAULT_LABEL_ALPHA = 0.5

CAPTION_FONT_SIZE = 20
CAPTION_PADDING = 8

# Raw label --> RGB color, e.g. for gtFine labelIds.
_RAW_LABEL_COLORS = np.full((MAX_LABEL_COUNT, 3), IGNORE_COLOR, dtype=np.uint8)
for _raw_id, _color in CITYSCAPES_CLASS_COLORS.items():
    _RAW_LABEL_COLORS[_raw_id] = _color


def overlay_label(
    image: np.ndarray, label: np.ndarray, alpha: float = DEFAULT_LABEL_ALPHA
) -> np.ndarray:
    """
    Blend a (H, W) raw label's colors over a (H, W, 3) uint8 image.
    """

    colors = _RAW_LABEL_COLORS[label].astype(np.float32)
    blended = image.astype(np.float32) * (1 - alpha) + colors * alpha
    return blended.round().astype(np.uint8)


@functools.cache
def _get_caption_font() -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=CAPTION_FONT_SIZE)


def add_caption(image: np.ndarray, text: str) -> np.ndarray:
    """
    Draw `text` in the top left of a (H, W, 3) uint8 image, on a black box.
    """

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    font = _get_caption_font()

    position = (CAPTION_PADDING, CAPTION_PADDING)
    left, top, right, bottom = draw.textbbox(position, text, font=font)
    half_padding = CAPTION_PADDING // 2
    draw.rectangle(
        (
            left - half_padding,
            top - half_padding,
            right + half_padding,
            bottom + half_padding,
        ),
        fill=(0, 0, 0),
    )
    draw.text(position, text, fill=(255, 255, 255), font=font)

    return np.asarray(pil_image)


class _VideoWriter:
    """
    Write (H, W, 3) uint8 frames to an MP4 with ffmpeg, so that `path` never
    holds a partial file.
    """

    def __init__(self, path: Path, height: int, width: int, fps: int):
        self.path = path
        self.temporary_path = path.with_name(f"{path.name}.tmp")
        self.height, self.width = height, width

        path.parent.mkdir(parents=True, exist_ok=True)
        # Raw RGB frames in, H.264 in yuv420p out, which most players support.
        # fmt: off
        command = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-f", "mp4",
            str(self.temporary_path),
        ]
        # fmt: on
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"{frame.shape=} must be {(self.height, self.width, 3)}."
            )
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.process.stdin.close()
        stderr = self.process.stderr.read().decode()
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for '{self.path}': {stderr}")
        self.temporary_path.replace(self.path)

    def abort(self) -> None:
        self.process.kill()
        self.process.wait()
        self.temporary_path.unlink(missing_ok=True)


def _write_city_video(
    path: Path,
    split: str,
    frames: list[CityscapesDatapoint],
    fps: int,
    labeled_frame_hold_count: int,
    worker_count: int,
) -> None:
    """
    Write `frames` as a video, overlaying labels & holding labeled frames.
    """

    start = time.perf_counter()

    writer = None
    try:
        loaded_frames = load_in_parallel(frames, worker_count)
        for dp, loaded in zip(frames, loaded_frames, strict=True):
            if writer is None:
                writer = _VideoWriter(path, *loaded.image.shape[:2], fps)

            caption = (
                f"{split} | {dp.city} | clip {dp.clip_idx} | "
                f"frame {dp.frame_idx}"
            )
            if loaded.label is None:
                writer.write(add_caption(loaded.image, caption))
                continue

            frame = add_caption(
                overlay_label(loaded.image, loaded.label),
                f"{caption} | fine label",
            )
            for _ in range(labeled_frame_hold_count):
                writer.write(frame)

        if writer is not None:
            writer.close()
    except BaseException:
        if writer is not None:
            writer.abort()
        raise

    logger.info(
        f"Wrote '{path}': {len(frames):,} frames, "
        f"{sum(dp.label_path is not None for dp in frames):,} labeled, in "
        f"{time.perf_counter() - start:.0f}s."
    )


def write_cityscapes_videos(
    image_dir: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
    splits: tuple[str, ...] = ("train", "val"),  # IS-OOS splits to write.
    labeled_splits: tuple[str, ...] = ("train", "val"),  # Splits to overlay.
    fps: int = DEFAULT_FPS,
    labeled_frame_hold_seconds: float = DEFAULT_LABELED_FRAME_HOLD_SECONDS,
    force: bool = False,  # Overwrite existing files, instead of skipping them.
    worker_count: int = DEFAULT_WORKER_COUNT,
) -> None:
    """
    Write a video per (split, city) of every sequence frame, in clip & frame
    order, with fine labels overlaid on labeled frames.

    Videos are written to <output_dir>/<split>/<city>.mp4.
    """

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    labeled_frame_hold_count = max(1, round(fps * labeled_frame_hold_seconds))

    for split in splits:
        # (city, clip index, frame index) --> label path.
        frame_to_label_path: dict[tuple[str, str, str], Path] = {}
        if split in labeled_splits:
            labeled_dataset = CityscapesLabeledDataset(
                image_dir=image_dir,
                label_dir=label_dir,
                selected_is_oos_splits=[split],
            )
            for dp in labeled_dataset.get_datapoints():
                key = (dp.city, dp.clip_idx, dp.frame_idx)
                frame_to_label_path[key] = dp.label_path

        city_to_frames: dict[str, list[CityscapesDatapoint]] = (
            collections.defaultdict(list)
        )
        unlabeled_dataset = CityscapesUnlabeledDataset(
            image_dir=image_dir,
            selected_is_oos_splits=[split],
            keep_every_nth_frame=1,
        )
        for dp in unlabeled_dataset.get_datapoints():
            label_path = frame_to_label_path.get(
                (dp.city, dp.clip_idx, dp.frame_idx)
            )
            city_to_frames[dp.city].append(
                CityscapesDatapoint(
                    image_path=dp.image_path, label_path=label_path
                )
            )

        matched_count = sum(
            dp.label_path is not None
            for frames in city_to_frames.values()
            for dp in frames
        )
        if matched_count != len(frame_to_label_path):
            logger.warning(
                f"Only {matched_count:,}/{len(frame_to_label_path):,} {split} "
                "labels have a frame in the sequence images."
            )

        for city, frames in sorted(city_to_frames.items()):
            path = output_dir / split / f"{city}.mp4"
            if path.exists() and not force:
                logger.info(f"Skipping '{path}', which exists.")
                continue

            _write_city_video(
                path,
                split,
                frames,
                fps,
                labeled_frame_hold_count,
                worker_count,
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    write_cityscapes_videos(
        image_dir=CITYSCAPES_DIR / "leftImg8bit_sequence_2x_downsampled",
        label_dir=CITYSCAPES_DIR / "gtFine_2x_downsampled",
        output_dir=CITYSCAPES_DIR
        / "leftImg8bit_sequence_2x_downsampled_videos",
    )
