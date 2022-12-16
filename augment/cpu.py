import torch
import random


class RandomSwapLR(object):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        assert 0 <= p <= 1, "invalid probability value"
        self.p = p

    def __call__(self, stems: torch.Tensor):
        """
        Args:
            stems (torch.Tensor): (Num_sources, Num_channels, L)
        Return:
            stems (torch.Tensor): (Num_sources, Num_channels, L)
        """
        tmp = torch.flip(stems, [1])
        for i in range(stems.shape[0]):
            if random.random() < self.p:
                stems[i] = tmp[i]
        return stems


class RandomGain(object):
    def __init__(self, low=0.25, high=1.25) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def __call__(self, stems):
        """
        Args:
            stems (torch.Tensor): (Num_sources, Num_channels, L)
        Return:
            stems (torch.Tensor): (Num_sources, Num_channels, L)
        """
        gains = torch.rand(
            stems.shape[0], 1, 1, device=stems.device) * (self.high - self.low) + self.low
        stems = stems * gains
        return stems


class RandomFlipPhase(RandomSwapLR):
    def __call__(self, stems: torch.Tensor):
        """
        Args:
            stems (torch.Tensor): (Num_sources, Num_channels, L)
        Return:
            stems (torch.Tensor): (Num_sources, Num_channels, L)
        """
        for i in range(stems.shape[0]):
            if random.random() < self.p:
                stems[i] *= -1
        return stems
