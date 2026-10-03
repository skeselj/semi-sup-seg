"""
Models for the segmentation task: image --> per-pixel classes.
"""

from .random_segmenter import RandomSegmenter
from .unet import UNet

__all__ = ["RandomSegmenter", "UNet"]
