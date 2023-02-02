import torch
import numpy as np

from typing import Optional

from asteroid.asteroid import complex_nn
from asteroid_filterbanks.transforms import from_torch_complex
from asteroid.asteroid.models.dcunet import BaseDCUNet
from asteroid.asteroid.masknn.base import BaseDCUMaskNet
from asteroid.asteroid.masknn.convolutional import DCUMaskNet
from asteroid.asteroid.utils.torch_utils import pad_x_to_y
from asteroid.asteroid.masknn._dcunet_architectures import (
    make_unet_encoder_decoder_args,
)


DCUNET_ARCHITECTURES = {
    "DCUNet-10": make_unet_encoder_decoder_args(
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding)
        (
            (2, 32, (7, 5), (2, 2), "auto"),
            (32, 64, (7, 5), (2, 2), "auto"),
            (64, 64, (5, 3), (2, 2), "auto"),
            (64, 64, (5, 3), (2, 2), "auto"),
            (64, 64, (5, 3), (2, 1), "auto"),
        ),
        # Decoders: automatic inverse
        "auto",
    ),
    "DCUNet-16": make_unet_encoder_decoder_args(
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding)
        (
            (2, 32, (7, 5), (2, 2), "auto"),
            (32, 32, (7, 5), (2, 1), "auto"),
            (32, 64, (7, 5), (2, 2), "auto"),
            (64, 64, (5, 3), (2, 1), "auto"),
            (64, 64, (5, 3), (2, 2), "auto"),
            (64, 64, (5, 3), (2, 1), "auto"),
            (64, 64, (5, 3), (2, 2), "auto"),
            (64, 64, (5, 3), (2, 1), "auto"),
        ),
        # Decoders: automatic inverse
        "auto",
    ),
    "DCUNet-20": make_unet_encoder_decoder_args(
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding)
        (
            (2, 32, (7, 1), (1, 1), "auto"),
            (32, 32, (1, 7), (1, 1), "auto"),
            (32, 64, (7, 5), (2, 2), "auto"),
            (64, 64, (7, 5), (2, 1), "auto"),
            (64, 64, (5, 3), (2, 2), "auto"),
            (64, 64, (5, 3), (2, 1), "auto"),
            (64, 64, (5, 3), (2, 2), "auto"),
            (64, 64, (5, 3), (2, 1), "auto"),
            (64, 64, (5, 3), (2, 2), "auto"),
            (64, 90, (5, 3), (2, 1), "auto"),
        ),
        # Decoders: automatic inverse
        "auto",
    ),
    "Large-DCUNet-20": make_unet_encoder_decoder_args(
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding)
        (
            (2, 45, (7, 1), (1, 1), "auto"),
            (45, 45, (1, 7), (1, 1), "auto"),
            (45, 90, (7, 5), (2, 2), "auto"),
            (90, 90, (7, 5), (2, 1), "auto"),
            (90, 90, (5, 3), (2, 2), "auto"),
            (90, 90, (5, 3), (2, 1), "auto"),
            (90, 90, (5, 3), (2, 2), "auto"),
            (90, 90, (5, 3), (2, 1), "auto"),
            (90, 90, (5, 3), (2, 2), "auto"),
            (90, 128, (5, 3), (2, 1), "auto"),
        ),
        # Decoders:
        # (in_chan, out_chan, kernel_size, stride, padding, output_padding)
        (
            (128, 90, (5, 3), (2, 1), "auto", (0, 0)),
            (180, 90, (5, 3), (2, 2), "auto", (0, 0)),
            (180, 90, (5, 3), (2, 1), "auto", (0, 0)),
            (180, 90, (5, 3), (2, 2), "auto", (0, 0)),
            (180, 90, (5, 3), (2, 1), "auto", (0, 0)),
            (180, 90, (5, 3), (2, 2), "auto", (0, 0)),
            (180, 90, (7, 5), (2, 1), "auto", (0, 0)),
            (180, 90, (7, 5), (2, 2), "auto", (0, 0)),
            (135, 90, (1, 7), (1, 1), "auto", (0, 0)),
            (135, 1, (7, 1), (1, 1), "auto", (0, 0)),
        ),
    ),
    "mini": make_unet_encoder_decoder_args(
        # This is a dummy architecture used for Asteroid unit tests.
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding)
        (
            (2, 4, (7, 5), (2, 2), "auto"),
            (4, 8, (7, 5), (2, 2), "auto"),
            (8, 16, (5, 3), (2, 2), "auto"),
        ),
        # Decoders: automatic inverse
        "auto",
    ),
}


class OurDCUMaskNet(BaseDCUMaskNet):
    r"""Masking part of DCUNet, as proposed in [1].

    Valid `architecture` values for the ``default_architecture`` classmethod are:
    "Large-DCUNet-20", "DCUNet-20", "DCUNet-16", "DCUNet-10" and "mini".

    Valid `fix_length_mode` values are [None, "pad", "trim"].

    Input shape is expected to be $(batch, nfreqs, time)$, with $nfreqs - 1$ divisible
    by $f_0 * f_1 * ... * f_N$ where $f_k$ are the frequency strides of the encoders,
    and $time - 1$ is divisible by $t_0 * t_1 * ... * t_N$ where $t_N$ are the time
    strides of the encoders.

    References
        [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
        Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """

    _architectures = DCUNET_ARCHITECTURES

    def __init__(self, encoders, decoders, fix_length_mode=None, **kwargs):
        self.fix_length_mode = fix_length_mode
        self.encoders_stride_product = np.prod(
            [enc_stride for _, _, _, enc_stride, _ in encoders], axis=0
        )

        # Avoid circual import
        from asteroid.asteroid.masknn.convolutional import (
            DCUNetComplexDecoderBlock,
            DCUNetComplexEncoderBlock,
        )

        super().__init__(
            encoders=[DCUNetComplexEncoderBlock(*args) for args in encoders],
            decoders=[DCUNetComplexDecoderBlock(*args) for args in decoders[:-1]],
            output_layer=complex_nn.ComplexConvTranspose2d(*decoders[-1]),
            **kwargs,
        )

    def fix_input_dims(self, x):
        return _fix_dcu_input_dims(
            self.fix_length_mode, x, torch.from_numpy(self.encoders_stride_product)
        )

    def fix_output_dims(self, out, x):
        return _fix_dcu_output_dims(self.fix_length_mode, out, x)


class StereoDCUMaskNet(OurDCUMaskNet):
    _architectures = DCUNET_ARCHITECTURES

    def fix_input_dims(self, x):
        return _fix_dcu_input_dims(
            self.fix_length_mode, x, torch.from_numpy(self.encoders_stride_product)
        )

    # def fix_output_dims(self, out, x):
    #    return _fix_dcu_output_dims(self.fix_length_mode, out, x)


def _fix_dcu_input_dims(fix_length_mode: Optional[str], x, encoders_stride_product):
    """Pad or trim `x` to a length compatible with DCUNet."""
    freq_prod = int(encoders_stride_product[0])
    time_prod = int(encoders_stride_product[1])
    if (x.shape[-2] - 1) % freq_prod:
        raise TypeError(
            f"Input shape must be [batch, chs, freq + 1, time + 1] with freq divisible by "
            f"{freq_prod}, got {x.shape} instead"
        )
    time_remainder = (x.shape[-1] - 1) % time_prod
    if time_remainder:
        if fix_length_mode is None:
            raise TypeError(
                f"Input shape must be [batch, chs, freq + 1, time + 1] with time divisible by "
                f"{time_prod}, got {x.shape} instead. Set the 'fix_length_mode' argument "
                f"in 'DCUNet' to 'pad' or 'trim' to fix shapes automatically."
            )
        elif fix_length_mode == "pad":
            pad_shape = [0, time_prod - time_remainder]
            x = torch.nn.functional.pad(x, pad_shape, mode="constant")
        elif fix_length_mode == "trim":
            pad_shape = [0, -time_remainder]
            x = torch.nn.functional.pad(x, pad_shape, mode="constant")
        else:
            raise ValueError(f"Unknown fix_length mode '{fix_length_mode}'")
    return x


def _fix_dcu_output_dims(fix_length_mode: Optional[str], out, x):
    """Fix shape of `out` to the original shape of `x`."""
    return pad_x_to_y(out, x)


class StereoDCUNet(BaseDCUNet):
    masknet_class = StereoDCUMaskNet

    def apply_masks(self, tf_rep, est_masks):
        masked_tf_rep = est_masks * tf_rep
        return from_torch_complex(masked_tf_rep)


class DCUNetSplit(torch.nn.Module):
    def __init__(
        self,
        architecture: str,
        stft_n_filters: int = 4096,
        stft_kernel_size: int = 4096,
        stft_stride: int = 1024,
        sample_rate: float = 44100.0,
        num_sources: int = 4,
    ) -> None:
        super().__init__()
        self.separators = torch.nn.ModuleList()
        for _ in range(num_sources):
            self.separators.append(
                StereoDCUNet(
                    architecture,
                    stft_n_filters=stft_n_filters,
                    stft_kernel_size=stft_kernel_size,
                    stft_stride=stft_stride,
                    sample_rate=sample_rate,
                    fix_length_mode="pad",
                    n_src=2,
                )
            )

    def forward(self, x: torch.Tensor):
        y_hats = []
        for idx, separator in enumerate(self.separators):
            y_hats.append(separator(x))

        y_hat = torch.stack(y_hats, dim=1)

        return y_hat
