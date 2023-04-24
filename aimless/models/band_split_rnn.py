import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

v7_freqs = [
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    1250,
    1500,
    1750,
    2000,
    2250,
    2500,
    2750,
    3000,
    3250,
    3500,
    3750,
    4000,
    4500,
    5000,
    5500,
    6000,
    6500,
    7000,
    7500,
    8000,
    9000,
    10000,
    11000,
    12000,
    13000,
    14000,
    15000,
    16000,
    18000,
    20000,
    22000,
]


class BandSplitRNN(nn.Module):
    def __init__(
        self,
        n_fft: int,
        split_freqs: List[int] = v7_freqs,
        hidden_size: int = 128,
        num_layers: int = 12,
        norm_groups: int = 4,
    ) -> None:
        super().__init__()

        # get split freq bins index from split freqs
        index = [0] + [int(n_fft * f / 44100) for f in split_freqs] + [n_fft // 2 + 1]
        chunk_size = [index[i + 1] - index[i] for i in range(len(index) - 1)]
        self.split_sections = tuple(chunk_size)

        # stage 1: band split modules
        self.norm1_list = nn.ModuleList(
            [nn.LayerNorm(chunk_size[i]) for i in range(len(chunk_size))]
        )
        self.fc1_list = nn.ModuleList(
            [nn.Linear(chunk_size[i], hidden_size) for i in range(len(chunk_size))]
        )

        # stage 2: RNN modules
        self.band_lstms = nn.ModuleList()
        self.time_lstms = nn.ModuleList()
        self.band_group_norms = nn.ModuleList()
        self.time_group_norms = nn.ModuleList()
        for i in range(num_layers):
            self.band_group_norms.append(nn.GroupNorm(norm_groups, hidden_size))
            self.band_lstms.append(
                nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                    num_layers=1,
                    bidirectional=True,
                    proj_size=hidden_size // 2,
                )
            )

            self.time_lstms.append(
                nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                    num_layers=1,
                    bidirectional=True,
                    proj_size=hidden_size // 2,
                )
            )
            self.time_group_norms.append(nn.GroupNorm(norm_groups, hidden_size))

        # stage 3: band merge modules and mask prediction modules
        self.norm2_list = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for i in range(len(chunk_size))]
        )
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.Tanh(),
                    nn.Linear(hidden_size * 4, chunk_size[i]),
                )
                for i in range(len(chunk_size))
            ]
        )

    def forward(self, mag: torch.Tensor):
        log_mag = torch.log(mag + 1e-8)
        batch, channels, freq_bins, time_bins = log_mag.shape
        # merge channels to batch
        log_mag = log_mag.view(-1, freq_bins, time_bins).transpose(1, 2)

        # stage 1: band split modules
        tmp = []
        for fc, norm, sub_band in zip(
            self.fc1_list, self.norm1_list, log_mag.split(self.split_sections, dim=2)
        ):
            tmp.append(fc(norm(sub_band)))

        x = torch.stack(tmp, dim=1)

        # stage 2: RNN modules
        for band_lstm, time_lstm, band_norm, time_norm in zip(
            self.band_lstms,
            self.time_lstms,
            self.band_group_norms,
            self.time_group_norms,
        ):
            x_reshape = x.reshape(-1, x.shape[-2], x.shape[-1])
            x_reshape = time_norm(x_reshape.transpose(1, 2)).transpose(1, 2)
            x = time_lstm(x_reshape)[0].view(*x.shape) + x

            x_reshape = x.transpose(1, 2).reshape(-1, x.shape[-3], x.shape[-1])
            x_reshape = band_norm(x_reshape.transpose(1, 2)).transpose(1, 2)
            x = (
                band_lstm(x_reshape)[0]
                .view(x.shape[0], x.shape[2], x.shape[1], x.shape[3])
                .transpose(1, 2)
                + x
            )

        # stage 3: band merge modules and mask prediction modules
        tmp = []
        for i, (mlp, norm) in enumerate(zip(self.mlps, self.norm2_list)):
            tmp.append(mlp(norm(x[:, i])))

        mask = (
            torch.cat(tmp, dim=-1)
            .transpose(1, 2)
            .reshape(batch, channels, freq_bins, time_bins)
            .sigmoid()
        )
        return mask


class BandSplitRNNMulti(nn.Module):
    def __init__(
        self,
        n_fft: int,
        n_sources: int,
        split_freqs: List[int] = v7_freqs,
        hidden_size: int = 128,
        num_layers: int = 12,
        norm_groups: int = 4,
    ) -> None:
        super().__init__()

        # get split freq bins index from split freqs
        index = [0] + [int(n_fft * f / 44100) for f in split_freqs] + [n_fft // 2 + 1]
        chunk_size = [index[i + 1] - index[i] for i in range(len(index) - 1)]
        self.split_sections = tuple(chunk_size)

        self.n_sources = n_sources

        # stage 1: band split modules
        self.norm1_list = nn.ModuleList(
            [nn.LayerNorm(chunk_size[i]) for i in range(len(chunk_size))]
        )
        self.fc1_list = nn.ModuleList(
            [nn.Linear(chunk_size[i], hidden_size) for i in range(len(chunk_size))]
        )

        # stage 2: RNN modules
        self.band_lstms = nn.ModuleList()
        self.time_lstms = nn.ModuleList()
        self.band_group_norms = nn.ModuleList()
        self.time_group_norms = nn.ModuleList()
        for i in range(num_layers):
            self.band_group_norms.append(nn.GroupNorm(norm_groups, hidden_size))
            self.band_lstms.append(
                nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                    num_layers=1,
                    bidirectional=True,
                    proj_size=hidden_size // 2,
                )
            )

            self.time_lstms.append(
                nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                    num_layers=1,
                    bidirectional=True,
                    proj_size=hidden_size // 2,
                )
            )
            self.time_group_norms.append(nn.GroupNorm(norm_groups, hidden_size))

        # stage 3: band merge modules and mask prediction modules
        self.norm2_list = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for i in range(len(chunk_size))]
        )
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.Tanh(),
                    nn.Linear(hidden_size * 4, chunk_size[i] * n_sources),
                )
                for i in range(len(chunk_size))
            ]
        )

    def forward(self, mag: torch.Tensor):
        log_mag = torch.log(mag + 1e-8)
        batch, channels, freq_bins, time_bins = log_mag.shape
        # merge channels to batch
        log_mag = log_mag.view(-1, freq_bins, time_bins).transpose(1, 2)

        # stage 1: band split modules
        tmp = []
        for fc, norm, sub_band in zip(
            self.fc1_list, self.norm1_list, log_mag.split(self.split_sections, dim=2)
        ):
            tmp.append(fc(norm(sub_band)))

        x = torch.stack(tmp, dim=1)

        # stage 2: RNN modules
        for band_lstm, time_lstm, band_norm, time_norm in zip(
            self.band_lstms,
            self.time_lstms,
            self.band_group_norms,
            self.time_group_norms,
        ):
            x_reshape = x.reshape(-1, x.shape[-2], x.shape[-1])
            x_reshape = time_norm(x_reshape.transpose(1, 2)).transpose(1, 2)
            x = time_lstm(x_reshape)[0].view(*x.shape) + x

            x_reshape = x.transpose(1, 2).reshape(-1, x.shape[-3], x.shape[-1])
            x_reshape = band_norm(x_reshape.transpose(1, 2)).transpose(1, 2)
            x = (
                band_lstm(x_reshape)[0]
                .view(x.shape[0], x.shape[2], x.shape[1], x.shape[3])
                .transpose(1, 2)
                + x
            )

        # stage 3: band merge modules and mask prediction modules
        tmp = []
        for i, (mlp, norm) in enumerate(zip(self.mlps, self.norm2_list)):
            tmp.append(
                mlp(norm(x[:, i])).view(x.shape[0], x.shape[2], self.n_sources, -1)
            )

        mask = (
            torch.cat(tmp, dim=-1)
            .reshape(batch, channels, time_bins, self.n_sources, freq_bins)
            .permute(0, 3, 1, 4, 2)
            .softmax(dim=1)
        )
        return mask
