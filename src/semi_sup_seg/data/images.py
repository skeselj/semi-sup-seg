"""
Module to support working with images.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """
    Load a (H, W, 3) uint8 RGB image.
    """

    return np.asarray(Image.open(path).convert("RGB"))


def load_label(path: str | Path) -> np.ndarray:
    """
    Load a (H, W) uint8 label image.
    """

    return np.asarray(Image.open(path))


def get_image_size(path: str | Path) -> tuple[int, int]:
    """
    Get the (H, W) of an image file, without loading its pixels.
    """

    width, height = Image.open(path).size
    return height, width


def write_png_atomically(array: np.ndarray, path: str | Path) -> None:
    """
    Write `array` as a PNG, so that `path` never holds a partial file.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(f"{path.name}.tmp")
    Image.fromarray(array).save(temporary_path, format="PNG")
    temporary_path.replace(path)
