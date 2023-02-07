import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import TimeStretch, Spectrogram, InverseSpectrogram, Resample
from torchaudio import functional as aF
from torch_fftconv import fft_conv1d
from pathlib import Path

__all__ = ["SpeedPerturb", "RandomPitch", "RandomConvolutions", "CudaBase"]


class CudaBase(nn.Module):
    def __init__(self, rand_size, p=0.2):
        super().__init__()
        self.p = p
        self.rand_size = rand_size

    def _transform(self, stems, index):
        """
        Args:
            stems (torch.Tensor): (B, Num_channels, L)
            index (int): index of random transform
        Return:
            perturbed_stems (torch.Tensor): (B, Num_channels, L')
        """
        raise NotImplementedError

    def forward(self, stems: torch.Tensor):
        """
        Args:
            stems (torch.Tensor): (B, Num_sources, Num_channels, L)
        Return:
            perturbed_stems (torch.Tensor): (B, Num_sources, Num_channels, L')
        """
        shape = stems.shape
        orig_len = shape[-1]
        stems = stems.view(-1, *shape[-2:])
        select_mask = torch.rand(stems.shape[0], device=stems.device) < self.p
        if not torch.any(select_mask):
            return stems.view(*shape)

        select_idx = torch.where(select_mask)[0]
        perturbed_stems = torch.zeros_like(stems)
        perturbed_stems[~select_mask] = stems[~select_mask]
        selected_stems = stems[select_mask]
        rand_idx = torch.randint(
            self.rand_size, (selected_stems.shape[0],), device=stems.device
        )

        for i in range(self.rand_size):
            mask = rand_idx == i
            if not torch.any(mask):
                continue
            masked_stems = selected_stems[mask]
            perturbed_audio = self._transform(masked_stems, i).to(perturbed_stems.dtype)

            diff = perturbed_audio.shape[-1] - orig_len

            put_idx = select_idx[mask]
            if diff >= 0:
                perturbed_stems[put_idx] = perturbed_audio[..., :orig_len]
            else:
                perturbed_stems[put_idx, :, : orig_len + diff] = perturbed_audio

        perturbed_stems = perturbed_stems.view(*shape)
        return perturbed_stems


class SpeedPerturb(CudaBase):
    def __init__(self, orig_freq=44100, speeds=[90, 100, 110], **kwargs):
        super().__init__(len(speeds), **kwargs)
        self.orig_freq = orig_freq
        self.resamplers = nn.ModuleList()
        self.speeds = speeds
        for s in self.speeds:
            new_freq = self.orig_freq * s // 100
            self.resamplers.append(Resample(self.orig_freq, new_freq))

    def _transform(self, stems, index):
        y = self.resamplers[index](stems.view(-1, stems.shape[-1])).view(
            *stems.shape[:-1], -1
        )
        return y


class RandomPitch(CudaBase):
    def __init__(
        self, semitones=[-2, -1, 0, 1, 2], n_fft=2048, hop_length=512, **kwargs
    ):
        super().__init__(len(semitones), **kwargs)
        self.resamplers = nn.ModuleList()

        semitones = torch.tensor(semitones, dtype=torch.float32)
        rates = 2 ** (-semitones / 12)
        rrates = rates.reciprocal()
        rrates = (rrates * 100).long()
        rrates[rrates % 2 == 1] += 1
        rates = 100 / rrates

        self.register_buffer("rates", rates)
        self.spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
        self.inv_spec = InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)
        self.stretcher = TimeStretch(hop_length, n_freq=n_fft // 2 + 1)

        for rr in rrates.tolist():
            self.resamplers.append(Resample(rr, 100))

    def _transform(self, stems, index):
        spec = self.spec(stems)
        stretched_spec = self.stretcher(spec, self.rates[index])
        stretched_stems = self.inv_spec(stretched_spec)
        shifted_stems = self.resamplers[index](
            stretched_stems.view(-1, stretched_stems.shape[-1])
        ).view(*stretched_stems.shape[:-1], -1)
        return shifted_stems


class RandomConvolutions(CudaBase):
    def __init__(self, target_sr: int, ir_folder: str, **kwargs):
        ir_folder = Path(ir_folder)
        ir_files = list(ir_folder.glob("**/*.wav"))
        impulses = []
        for ir_file in ir_files:
            ir, sr = torchaudio.load(ir_file)
            if ir.shape[0] > 2:
                continue
            if sr != target_sr:
                ir = aF.resample(ir, sr, target_sr)
            if ir.shape[0] == 1:
                ir = ir.repeat(2, 1)
            impulses.append(ir)

        super().__init__(len(impulses), **kwargs)
        for i, impulse in enumerate(impulses):
            self.register_buffer(f"impulse_{i}", impulse)

    def _transform(self, stems, index):
        ir = self.get_buffer(f"impulse_{index}").unsqueeze(1)
        ir_flipped = ir.flip(-1)
        padded_stems = F.pad(stems, (ir.shape[-1] - 1, 0))
        # TODO: dynamically use F.conv1d if impulse is short
        convolved_stems = fft_conv1d(padded_stems, ir_flipped, groups=2)
        return convolved_stems
