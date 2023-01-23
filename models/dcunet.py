import torch
from models.asteroid.asteroid.models.dcunet import DCUNet


class FixedDCUNet(torch.nn.Module):
    def __init__(
        self,
        architecture: str,
        stft_n_filters: int,
        stft_kernel_size: int,
        stft_stride: int,
        sample_rate: float,
        fix_length_mode: str,
    ) -> None:
        super().__init__()
        self.model = DCUNet(
            architecture=architecture,
            stft_n_filters=stft_n_filters,
            stft_kernel_size=stft_kernel_size,
            stft_stride=stft_stride,
            sample_rate=sample_rate,
            fix_length_mode=fix_length_mode,
        )

    def forward(self, x: torch.Tensor):
        print(x.shape)
        return self.model(x)
