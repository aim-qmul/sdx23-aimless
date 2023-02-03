import torch
import random
import numpy as np
import pyloudnorm as pyln
from pedalboard import Limiter, Pedalboard, Gain
from typing import List, Tuple

__all__ = ["RandomSwapLR", "RandomGain", "RandomFlipPhase", "LimitAug", "CPUBase"]


class CPUBase(object):
    def __call__(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (Tuple[torch.Tensor, torch.Tensor]): (mixture, stems) where
                mixture : (Num_channels, L)
                stems: (Num_sources, Num_channels, L)
        Return:
            Tuple[torch.Tensor, torch.Tensor]: (mixture, stems) where
                mixture: (Num_channels, L)
                stems: (Num_sources, Num_channels, L)
        """
        mixture, stems = x
        stems = self._transform(stems)
        return (stems.sum(0), stems)

    def _transform(self, stems: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RandomSwapLR(CPUBase):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        assert 0 <= p <= 1, "invalid probability value"
        self.p = p

    def _transform(self, stems: torch.Tensor):
        tmp = torch.flip(stems, [1])
        for i in range(stems.shape[0]):
            if random.random() < self.p:
                stems[i] = tmp[i]
        return stems


class RandomGain(CPUBase):
    def __init__(self, low=0.25, high=1.25) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def _transform(self, stems):
        gains = (
            torch.rand(stems.shape[0], 1, 1, device=stems.device)
            * (self.high - self.low)
            + self.low
        )
        stems = stems * gains
        return stems


class RandomFlipPhase(RandomSwapLR):
    def _transform(self, stems: torch.Tensor):
        for i in range(stems.shape[0]):
            if random.random() < self.p:
                stems[i] *= -1
        return stems


def db2linear(x):
    return 10 ** (x / 20)


class LimitAug(CPUBase):
    def __init__(
        self,
        target_lufs_mean=-10.887,
        target_lufs_std=1.191,
        target_loudnorm_lufs=-14.0,
        max_release_ms=200.0,
        min_release_ms=30.0,
        sample_rate=44100,
    ) -> None:
        """
        Args:
            target_lufs_mean (float): mean of target LUFS. default: -10.887 (corresponding to the statistics of musdb-L)
            target_lufs_std (float): std of target LUFS. default: 1.191 (corresponding to the statistics of musdb-L)
            target_loudnorm_lufs (float): target LUFS after loudnorm. default: -14.0
            max_release_ms (float): max release time of limiter. default: 200.0
            min_release_ms (float): min release time of limiter. default: 30.0
            sample_rate (int): sample rate of audio. default: 44100
        """
        super().__init__()
        self.target_lufs_sampler = torch.distributions.Normal(
            target_lufs_mean, target_lufs_std
        )
        self.target_loudnorm_lufs = target_loudnorm_lufs
        self.sample_rate = sample_rate
        self.board = Pedalboard([Gain(0), Limiter(threshold_db=0.0, release_ms=100.0)])
        self.limiter_release_sampler = torch.distributions.Uniform(
            min_release_ms, max_release_ms
        )
        self.meter = pyln.Meter(sample_rate)

    def __call__(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mixture, stems = x
        mixture_np = mixture.numpy().T
        loudness = self.meter.integrated_loudness(mixture_np)
        target_lufs = self.target_lufs_sampler.sample().item()
        self.board[1].release_ms = self.limiter_release_sampler.sample().item()

        if np.isinf(loudness):
            aug_gain = 0.0
        else:
            aug_gain = target_lufs - loudness
        self.board[0].gain_db = aug_gain

        new_mixture_np = self.board(mixture_np, self.sample_rate)
        after_loudness = self.meter.integrated_loudness(new_mixture_np)

        if not np.isinf(after_loudness):
            target_gain = self.target_loudnorm_lufs - after_loudness
            new_mixture_np *= db2linear(target_gain)

        new_mixture = torch.tensor(new_mixture_np.T, dtype=mixture.dtype)
        # apply element-wise gain to stems
        stems *= new_mixture.abs() / mixture.abs().add(1e-8)
        return (new_mixture, stems)
