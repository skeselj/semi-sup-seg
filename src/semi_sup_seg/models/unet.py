"""
Module to support defining U-Net models in PyTorch.
"""

import logging

import torch
import torch.nn.functional as F
from profiler import torch_phase
from torch import nn

from semi_sup_seg.constants import IMAGE_CHANNEL_COUNT, MAX_LABEL_COUNT

logger = logging.getLogger(__name__)


def _fan_in(layer: nn.Conv2d | nn.ConvTranspose2d) -> int:
    """
    Get number of input pixels contributing to each output pixel of `layer`.

    Regular convolution and de-convolution are supported.
    In both cases, the output of this function is:
          (# of pixels in operation receptive field)
        x (# of input channels)
    """

    kernel_height, kernel_width = layer.kernel_size

    if isinstance(layer, nn.Conv2d):
        return kernel_height * kernel_width * layer.in_channels
    elif isinstance(layer, nn.ConvTranspose2d):
        stride_height, stride_width = layer.stride
        return (
            (kernel_height // stride_height)
            * (kernel_width // stride_width)
            * layer.in_channels
        )

    raise NotImplementedError(f"_fan_in does not support {type(layer)}")


def _init_conv(
    layer: nn.Conv2d | nn.ConvTranspose2d, scale: float = 1.0
) -> None:
    """
    Initialize weights from N(0, scale**2 / fan-in), and biases to 0.
    """

    std = scale * (1 / _fan_in(layer)) ** 0.5
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    nn.init.zeros_(layer.bias)


class _ConvOp(nn.Module):
    """
    The enhanced convolution operation used in this module.

    Two convolutional layers with a skip connection at the end.
    """

    def __init__(
        self,
        input_channel_count: int,
        output_channel_count: int,
        conv_kernel_size: int,
        conv_padding: int,
    ):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=input_channel_count,
            out_channels=output_channel_count,
            kernel_size=conv_kernel_size,
            padding=conv_padding,
        )
        self.conv_2 = nn.Conv2d(
            in_channels=output_channel_count,
            out_channels=output_channel_count,
            kernel_size=conv_kernel_size,
            padding=conv_padding,
        )
        self.activation = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize learnable parameters.
        """

        _init_conv(self.conv_1)
        _init_conv(self.conv_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations_1 = self.activation(self.conv_1(x))
        activations_2 = self.activation(self.conv_2(activations_1))
        return activations_1 + activations_2


class _DownSampleOp(nn.Module):
    """
    The downsample operation used in this module.

    A 2x2 max pool. Every 2x2 patch becomes its largest pixel.
    """

    def __init__(self):
        super().__init__()
        self.op = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class _UpSampleOp(nn.Module):
    """
    The upsample operation used in this module.

    A 2x2 transposed convolution with stride 2 ("deconv"). Every pixel becomes
    a learned 2x2 patch.
    """

    def __init__(self, input_channel_count: int, output_channel_count: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(
            in_channels=input_channel_count,
            out_channels=output_channel_count,
            kernel_size=2,
            stride=2,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize learnable parameters.
        """

        _init_conv(self.op)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class UNet(nn.Module):
    """
    Basic implementation of the U-Net architecture for per-pixel classification.
    """

    DEFAULT_BASE_CHANNEL_COUNT = 32
    DEFAULT_LEVEL_COUNT = 5
    BRIDGE_CONV_SIZE = 3
    DEFAULT_CONV_SIZES = (3, 3, 5, 5, 5)  # Per-level.

    @classmethod
    def get_conv_kernel_sizes(
        cls, conv_kernel_sizes: tuple[int, ...] | None, level_count: int
    ) -> tuple[int, ...]:
        """
        Get validated per-level conv kernel sizes, filling in the default.
        """

        if conv_kernel_sizes is None:
            if level_count != len(cls.DEFAULT_CONV_SIZES):
                raise ValueError(
                    f"{cls.DEFAULT_CONV_SIZES=} is for "
                    f"{len(cls.DEFAULT_CONV_SIZES)} levels, so set "
                    f"conv_kernel_sizes for {level_count=}."
                )
            return cls.DEFAULT_CONV_SIZES

        conv_kernel_sizes = tuple(conv_kernel_sizes)
        if len(conv_kernel_sizes) != level_count:
            raise ValueError(
                f"{conv_kernel_sizes=} must have one size per level, "
                f"{level_count=}."
            )
        if any(size < 1 or size % 2 == 0 for size in conv_kernel_sizes):
            raise ValueError(f"{conv_kernel_sizes=} must all be odd.")

        return conv_kernel_sizes

    def __init__(
        self,
        *,
        base_height: int,
        base_width: int,
        output_channel_count: int,
        input_channel_count: int = IMAGE_CHANNEL_COUNT,
        base_channel_count: int = DEFAULT_BASE_CHANNEL_COUNT,
        level_count: int = DEFAULT_LEVEL_COUNT,
        conv_kernel_sizes: tuple[int, ...] | None = None,
    ):
        """
        Construct the model.

        Parameters
        ----------
            base_height
            base_width: input images are resized to (base_height x base_width)
                before processing. After processing, outputs are resized to the
                resolution of the original input image.
            output_channel_count: number of channels in the model output, one
                per class. Must be in [2, MAX_LABEL_COUNT].
            input_channel_count: number of channels in the model input.
            base_channel_count: number of channels in the output of the 1st
                level of processing.
            level_count: number of levels of processing.
            conv_kernel_sizes: per level, the (odd) kernel size of its encoder
                & decoder convs. None means DEFAULT_CONV_SIZES.
        """

        super().__init__()

        if not 2 <= output_channel_count <= MAX_LABEL_COUNT:
            raise ValueError(
                f"{output_channel_count=} must be in [2, {MAX_LABEL_COUNT}]."
            )

        downsample_factor = 2 ** (level_count - 1)
        if base_height % downsample_factor or base_width % downsample_factor:
            raise ValueError(
                f"{base_height=} x {base_width=} must both be divisible by "
                f"{downsample_factor=}"
            )

        self.base_height = base_height
        self.base_width = base_width
        self.output_channel_count = output_channel_count
        self.input_channel_count = input_channel_count
        self.base_channel_count = base_channel_count
        self.level_count = level_count
        self.conv_kernel_sizes = self.get_conv_kernel_sizes(
            conv_kernel_sizes, level_count
        )

        # Level --> number of output feature maps.
        channel_counts = [
            base_channel_count * 2**level for level in range(level_count)
        ]
        input_bridge_channel_count = base_channel_count // 2

        # Converts input image to feature maps processable by 1st level.
        self.input_bridge = nn.Conv2d(
            in_channels=input_channel_count,
            out_channels=input_bridge_channel_count,
            kernel_size=self.BRIDGE_CONV_SIZE,
            padding=self.BRIDGE_CONV_SIZE // 2,
        )

        # On the encoder side, a conv op turns C feature maps into 2*C maps.
        encoder_input_channel_counts = [
            input_bridge_channel_count,
            *channel_counts[:-1],
        ]
        self.encoder_conv_ops = nn.ModuleList(
            _ConvOp(
                input_channel_count=encoder_input_channel_counts[level],
                output_channel_count=channel_counts[level],
                conv_kernel_size=self.conv_kernel_sizes[level],
                conv_padding=self.conv_kernel_sizes[level] // 2,
            )
            for level in range(level_count)
        )
        # Down-sampling happens after all but the last encoder-side conv op.
        self.downsample_ops = nn.ModuleList(
            _DownSampleOp() for _ in range(level_count - 1)
        )

        # On the decoder side, a conv op turns C feature maps into C//2 maps.
        # Every level but the last has one.
        self.decoder_conv_ops = nn.ModuleList(
            _ConvOp(
                input_channel_count=2 * channel_counts[level],
                output_channel_count=channel_counts[level],
                conv_kernel_size=self.conv_kernel_sizes[level],
                conv_padding=self.conv_kernel_sizes[level] // 2,
            )
            for level in range(level_count - 1)
        )
        # Up-sampling happens before each decoder-side conv op.
        self.upsample_ops = nn.ModuleList(
            _UpSampleOp(
                input_channel_count=channel_counts[level + 1],
                output_channel_count=channel_counts[level],
            )
            for level in range(level_count - 1)
        )

        # Converts final feature maps into output per-class score maps.
        self.output_bridge = nn.Conv2d(
            in_channels=channel_counts[0],
            out_channels=output_channel_count,
            kernel_size=self.BRIDGE_CONV_SIZE,
            padding=self.BRIDGE_CONV_SIZE // 2,
        )

        self.reset_parameters()

        # Channels-last is the fastest layout for fp16 convolutions.
        self.to(memory_format=torch.channels_last)

        # fmt: off
        level_lines = []

        for level in range(level_count):
            shape = f"({channel_counts[level]:>4}, {base_height // 2**level:>4}, {base_width // 2**level:>4})"
            has_decoder = level < len(self.decoder_conv_ops)

            kernel_size = self.conv_kernel_sizes[level]
            level_lines.append(
                f"\tlevel {level}: "
                f"encoder {shape}, "
                f"decoder {shape if has_decoder else 'none'}, "
                f"{kernel_size}x{kernel_size} convs"
            )

        logger.info(
            "\n".join([
                f"Defined a U-Net over {level_count} levels, with (C, H, W) shaped feature maps.",
                *level_lines,
            ])
        )
        # fmt: on

    @property
    def config(self) -> dict[str, int | tuple[int, ...]]:
        """
        Get the arguments this model was constructed with.
        """

        return {
            "base_height": self.base_height,
            "base_width": self.base_width,
            "output_channel_count": self.output_channel_count,
            "input_channel_count": self.input_channel_count,
            "base_channel_count": self.base_channel_count,
            "level_count": self.level_count,
            "conv_kernel_sizes": self.conv_kernel_sizes,
        }

    def reset_parameters(self) -> None:
        """
        Initialize learnable parameters owned directly by this module.
        """

        _init_conv(self.input_bridge)
        # Small initial scores, so initial predictions are near-uniform.
        _init_conv(self.output_bridge, scale=0.01)

    @staticmethod
    def _resize(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if x.shape[-2:] == (height, width):
            return x

        # When shrinking, antialias: consider the receptive field, not just
        # the nearest 2x2.
        is_shrinking = height < x.shape[-2] or width < x.shape[-1]

        return F.interpolate(
            input=x,
            size=(height, width),
            mode="bilinear",
            antialias=is_shrinking,
        )

    def _normalized_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass on a (B, C, base_H, base_W) tensor.
        """

        with torch_phase("input_bridge"):
            x = self.input_bridge(x)

        # Encoder-side, from high level to low.
        skips = []
        for level, encoder_conv_op in enumerate(self.encoder_conv_ops):
            with torch_phase(f"encoder_{level}"):
                x = encoder_conv_op(x)

                if level < len(self.downsample_ops):
                    skips.append(x)
                    x = self.downsample_ops[level](x)

        # Decoder-side, from low level to high.
        for level in reversed(range(len(self.decoder_conv_ops))):
            with torch_phase(f"decoder_{level}"):
                x = self.upsample_ops[level](x)
                x = self.decoder_conv_ops[level](
                    torch.cat([x, skips[level]], dim=1)
                )

        with torch_phase("output_bridge"):
            return self.output_bridge(x)

    def forward(
        self, x: torch.Tensor, output_size: tuple[int, int] | None = None
    ) -> torch.Tensor:
        """
        Run the forward pass on a (B, C, H, W), of any H & W.

        Returns a (B, C_out, H, W) tensor of per-class scores, or a
        (B, C_out, *output_size) one if `output_size` is set.
        """

        output_height, output_width = output_size or x.shape[-2:]

        with torch_phase("resize_in"):
            x = self._resize(x, self.base_height, self.base_width)
            x = x.contiguous(memory_format=torch.channels_last)

        x = self._normalized_forward(x)

        with torch_phase("resize_out"):
            return self._resize(x, output_height, output_width)

    def compile(self, *args, **kwargs) -> None:
        """
        Compile the fixed-size portion of the forward pass.

        Overrides `nn.Module.compile`, which compiles all of `forward`.
        """

        self._normalized_forward = torch.compile(
            self._normalized_forward, *args, **kwargs
        )
