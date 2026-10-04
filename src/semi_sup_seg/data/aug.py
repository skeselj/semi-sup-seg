"""
Module to support data augmentation.
"""

import dataclasses
import math
from collections.abc import Sequence
from typing import Self

import torch
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class Augmentation:
    """
    Geometric augmentations of a batch of (image, label) pairs.

    Each field has shape (B,), with one value per batch element. Elements not
    given some augmentation have its identity value, e.g. zoom 1.

    The augmentations happen in field order, about the image center.
    """

    hflip: torch.Tensor  # bool.
    vflip: torch.Tensor  # bool.
    zoom: torch.Tensor  # > 1 magnifies.
    rotate_degrees: torch.Tensor  # > 0 rotates clockwise.
    translate_x: torch.Tensor  # Fraction of image width; > 0 moves right.
    translate_y: torch.Tensor  # Fraction of image height; > 0 moves down.

    @classmethod
    def cat(cls, augmentations: Sequence[Self]) -> Self:
        """
        Concatenate augmentations of several batches into one.
        """

        return cls(
            **{
                field.name: torch.cat(
                    [getattr(aug, field.name) for aug in augmentations]
                )
                for field in dataclasses.fields(cls)
            }
        )

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
        }

    def to_affine_matrices(self, height: int, width: int) -> torch.Tensor:
        """
        Get (B, 2, 3) matrices, as F.affine_grid takes.

        Each maps output coordinates to input coordinates, both in [-1, 1].
        """

        flip = torch.stack(
            [
                torch.where(self.hflip, -1.0, 1.0),
                torch.where(self.vflip, -1.0, 1.0),
            ],
            dim=1,
        )

        angle = torch.deg2rad(self.rotate_degrees)
        cos = torch.cos(angle) / self.zoom
        sin = torch.sin(angle) / self.zoom
        inverse_rotation = torch.stack(
            [torch.stack([cos, sin], dim=1), torch.stack([-sin, cos], dim=1)],
            dim=1,
        )

        pixel_linear = flip.unsqueeze(2) * inverse_rotation

        # Convert to coordinates in [-1, 1].
        scale = torch.tensor(
            [width, height], dtype=pixel_linear.dtype, device=flip.device
        )
        linear = pixel_linear * scale / scale.unsqueeze(1)

        # [-1, 1] spans the whole image, i.e. 2x the fraction.
        shift = 2 * torch.stack([self.translate_x, self.translate_y], dim=1)
        offset = -(linear @ shift.unsqueeze(2))

        return torch.cat([linear, offset], dim=2)

    def apply(
        self, images: torch.Tensor, labels: torch.Tensor, ignore_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply each element's augmentation to its (image, label) pair.

        Pixels sampled from outside the input are black in the image, and
        `ignore_id` in the label.

        Parameters
        ----------
            images: (B, C, H, W) float images.
            labels: (B, H, W) integer labels.
            ignore_id: label value of pixels excluded from the loss.
        """

        batch_size, channel_count, height, width = images.shape
        grid = F.affine_grid(
            self.to_affine_matrices(height, width),
            [batch_size, channel_count, height, width],
            align_corners=False,
        )

        images = F.grid_sample(
            images,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # Shift labels up by 1, so the zero padding marks out-of-input pixels.
        shifted_labels = F.grid_sample(
            (labels + 1).unsqueeze(1).float(),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        labels = shifted_labels.squeeze(1).long() - 1
        labels = labels.where(labels >= 0, ignore_id)

        return images, labels


@dataclasses.dataclass(frozen=True)
class AugmentationSampler:
    """
    Random geometric augmentations of (image, label) pairs.

    Each augmentation is applied to each batch element independently, with its
    probability. If applied, its parameter is sampled uniformly from its range.
    """

    hflip_prob: float = 0.5
    vflip_prob: float = 0.0

    translate_prob: float = 0.5
    max_translate: float = 0.1  # Fraction of image height / width.

    zoom_prob: float = 0.5
    max_zoom: float = 1.5  # Zoom in [1 / max_zoom, max_zoom], log-uniformly.

    rotate_prob: float = 0.5
    max_rotate_degrees: float = 10.0

    def sample(
        self, batch_size: int, device: torch.device | None = None
    ) -> Augmentation:
        """
        Sample an augmentation for each of `batch_size` elements.
        """

        def applied(prob: float) -> torch.Tensor:
            return torch.rand(batch_size, device=device) < prob

        def uniform(max_abs: float) -> torch.Tensor:
            return torch.empty(batch_size, device=device).uniform_(
                -max_abs, max_abs
            )

        translating = applied(self.translate_prob)
        return Augmentation(
            hflip=applied(self.hflip_prob),
            vflip=applied(self.vflip_prob),
            zoom=torch.exp(
                uniform(math.log(self.max_zoom)) * applied(self.zoom_prob)
            ),
            rotate_degrees=(
                uniform(self.max_rotate_degrees) * applied(self.rotate_prob)
            ),
            translate_x=uniform(self.max_translate) * translating,
            translate_y=uniform(self.max_translate) * translating,
        )

    def augment(
        self, images: torch.Tensor, labels: torch.Tensor, ignore_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, Augmentation]:
        """
        Sample an augmentation per (image, label) pair, and apply it.

        Returns the augmented images & labels, and the augmentation.
        """

        augmentation = self.sample(len(images), device=images.device)
        images, labels = augmentation.apply(images, labels, ignore_id)

        return images, labels, augmentation
