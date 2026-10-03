"""
Module to support defining U-Net models in PyTorch.
"""

import logging

import torch
import torch.nn.functional as F
from profiler import torch_phase
from torch import nn

from constants import IMAGE_CHANNEL_COUNT

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

        fan_in_1 = _fan_in(self.conv_1)
        nn.init.normal_(self.conv_1.weight, mean=0.0, std=(1 / fan_in_1) ** 0.5)
        nn.init.zeros_(self.conv_1.bias)

        fan_in_2 = _fan_in(self.conv_2)
        nn.init.normal_(self.conv_2.weight, mean=0.0, std=(1 / fan_in_2) ** 0.5)
        nn.init.zeros_(self.conv_2.bias)

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

        fan_in = _fan_in(self.op)
        nn.init.normal_(self.op.weight, mean=0.0, std=(1 / fan_in) ** 0.5)
        nn.init.zeros_(self.op.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class UNet(nn.Module):
    """
    Basic implementation of the U-Net architecture for per-pixel classification.
    """

    CONV_KERNEL_SIZE = 3
    CONV_PADDING = CONV_KERNEL_SIZE // 2

    DEFAULT_BASE_CHANNEL_COUNT = 32
    DEFAULT_LEVEL_COUNT = 5

    def __init__(
        self,
        *,
        base_height: int,
        base_width: int,
        output_channel_count: int,
        input_channel_count: int = IMAGE_CHANNEL_COUNT,
        base_channel_count: int = DEFAULT_BASE_CHANNEL_COUNT,
        level_count: int = DEFAULT_LEVEL_COUNT,
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
                per class. Must be in [2, 256].
            input_channel_count: number of channels in the model input.
            base_channel_count: number of channels in the output of the 1st
                level of processing.
            level_count: number of levels of processing.
        """

        super().__init__()

        if not 2 <= output_channel_count <= 256:
            raise ValueError(f"{output_channel_count=} must be in [2, 256].")

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

        # Level to number of output feature maps.
        level_to_channel_count = {
            "-1": base_channel_count // 2,  # Not a real level.
            **{
                str(level): base_channel_count * 2**level
                for level in range(level_count)
            },
        }

        # Converts input image to feature maps processable by 1st level.
        self.input_bridge = nn.Conv2d(
            in_channels=input_channel_count,
            out_channels=level_to_channel_count["-1"],
            kernel_size=self.CONV_KERNEL_SIZE,
            padding=self.CONV_PADDING,
        )

        # On the encoder side, a conv op turns C feature maps into 2*C maps.
        self.level_to_encoder_conv_op = nn.ModuleDict(
            {
                str(level): _ConvOp(
                    input_channel_count=level_to_channel_count[str(level - 1)],
                    output_channel_count=level_to_channel_count[str(level)],
                    conv_kernel_size=self.CONV_KERNEL_SIZE,
                    conv_padding=self.CONV_PADDING,
                )
                for level in range(level_count)
            }
        )
        # Down-sampling happens after most encoder-side conv ops.
        self.level_to_downsample_op = nn.ModuleDict(
            {str(level): _DownSampleOp() for level in range(level_count - 1)}
            | {str(level_count - 1): None}
        )

        # On the decoder side, a conv op turns C feature maps into C//2 maps.
        self.level_to_decoder_conv_op = nn.ModuleDict(
            {
                str(level): (
                    _ConvOp(
                        input_channel_count=(
                            2 * level_to_channel_count[str(level)]
                        ),
                        output_channel_count=level_to_channel_count[str(level)],
                        conv_kernel_size=self.CONV_KERNEL_SIZE,
                        conv_padding=self.CONV_PADDING,
                    )
                )
                for level in range(level_count - 1)
            }
            | {str(level_count - 1): None}
        )
        # Up-sampling happens before most decoder-side conv ops.
        self.level_to_upsample_op = nn.ModuleDict(
            {
                str(level): (
                    _UpSampleOp(
                        input_channel_count=level_to_channel_count[
                            str(level + 1)
                        ],
                        output_channel_count=level_to_channel_count[str(level)],
                    )
                )
                for level in range(level_count - 1)
            }
            | {str(level_count - 1): None}
        )

        # Converts final feature maps into output per-class score maps.
        self.output_bridge = nn.Conv2d(
            in_channels=level_to_channel_count["0"],
            out_channels=output_channel_count,
            kernel_size=self.CONV_KERNEL_SIZE,
            padding=self.CONV_PADDING,
        )

        self.reset_parameters()

        # fmt: off
        level_lines = []

        for level in range(level_count):
            shape = f"({level_to_channel_count[str(level)]:>4}, {base_height // 2**level:>4}, {base_width // 2**level:>4})"
            has_decoder = self.level_to_decoder_conv_op[str(level)] is not None

            level_lines.append(
                f"\tlevel {level}: "
                f"encoder {shape}, "
                f"decoder {shape if has_decoder else 'none'}"
            )

        logger.info(
            "\n".join([
                f"Defined a U-Net over {level_count} levels, with (C, H, W) shaped feature maps.",
                *level_lines,
            ])
        )
        # fmt: on

    @property
    def config(self) -> dict[str, int]:
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
        }

    def reset_parameters(self) -> None:
        """
        Initialize learnable parameters owned directly by this module.
        """

        fan_in = _fan_in(self.input_bridge)
        nn.init.normal_(
            self.input_bridge.weight, mean=0.0, std=(1 / fan_in) ** 0.5
        )
        nn.init.zeros_(self.input_bridge.bias)

        fan_in = _fan_in(self.output_bridge)
        nn.init.normal_(
            self.output_bridge.weight, mean=0.0, std=0.01 * (1 / fan_in) ** 0.5
        )
        nn.init.zeros_(self.output_bridge.bias)

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
        level_to_skip = {}
        for level in range(self.level_count):
            with torch_phase(f"encoder_{level}"):
                x = self.level_to_encoder_conv_op[str(level)](x)
                level_to_skip[str(level)] = x

                downsample_op = self.level_to_downsample_op[str(level)]
                if downsample_op is not None:
                    x = downsample_op(x)

        # Decoder-side, from low level to high.
        for level in reversed(range(self.level_count)):
            with torch_phase(f"decoder_{level}"):
                upsample_op = self.level_to_upsample_op[str(level)]
                if upsample_op is not None:
                    x = upsample_op(x)

                decoder_conv_op = self.level_to_decoder_conv_op[str(level)]
                if decoder_conv_op is not None:
                    x = decoder_conv_op(
                        torch.cat([x, level_to_skip[str(level)]], dim=1)
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

        x = self._normalized_forward(x)

        with torch_phase("resize_out"):
            return self._resize(x, output_height, output_width)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for a (B, C, H, W) tensor.

        Returns a (B, 1, H, W) tensor of class labels.
        """

        return self(x).argmax(dim=-3, keepdim=True).to(torch.uint8)

    def compile(self, *args, **kwargs) -> None:
        """
        Compile the fixed-size portion of the forward pass.

        Overrides `nn.Module.compile`, which compiles all of `forward`.
        """

        self._normalized_forward = torch.compile(
            self._normalized_forward, *args, **kwargs
        )
