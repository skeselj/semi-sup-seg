"""
Module to support running models in mixed precision.
"""

import torch

# Run forward passes in fp16 where safe (on CUDA only).
DEFAULT_MIXED_PRECISION = True


def autocast(
    device: torch.device, mixed_precision: bool = DEFAULT_MIXED_PRECISION
) -> torch.autocast:
    """
    Context in which eligible ops (e.g. convolutions) run in fp16.

    Has no effect unless `mixed_precision`, and `device` is a CUDA device.
    """

    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=mixed_precision and device.type == "cuda",
    )
