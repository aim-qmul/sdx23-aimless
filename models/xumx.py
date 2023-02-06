import torch
import torch.nn as nn
import torch.nn.functional as F


class X_UMX(nn.Module):
    __constants__ = ["max_bins"]

    max_bins: int

    def __init__(
        self, n_fft=4096, hidden_channels=512, max_bins=None, nb_channels=2, nb_layers=3
    ):
        super().__init__()

        self.nb_output_bins = n_fft // 2 + 1
        if max_bins:
            self.max_bins = max_bins
        else:
            self.max_bins = self.nb_output_bins
        self.hidden_channels = hidden_channels
        self.n_fft = n_fft
        self.nb_channels = nb_channels
        self.nb_layers = nb_layers

        self.input_means = nn.Parameter(torch.zeros(4 * self.max_bins))
        self.input_scale = nn.Parameter(torch.ones(4 * self.max_bins))

        self.output_means = nn.Parameter(torch.zeros(4 * self.nb_output_bins))
        self.output_scale = nn.Parameter(torch.ones(4 * self.nb_output_bins))

        self.affine1 = nn.Sequential(
            nn.Conv1d(
                nb_channels * self.max_bins * 4,
                hidden_channels * 4,
                1,
                bias=False,
                groups=4,
            ),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.Tanh(),
        )
        self.bass_lstm = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=nb_layers,
            dropout=0.4,
            bidirectional=True,
        )
        self.drums_lstm = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=nb_layers,
            dropout=0.4,
            bidirectional=True,
        )
        self.vocals_lstm = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=nb_layers,
            dropout=0.4,
            bidirectional=True,
        )
        self.other_lstm = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=nb_layers,
            dropout=0.4,
            bidirectional=True,
        )

        self.affine2 = nn.Sequential(
            nn.Conv1d(hidden_channels * 2, hidden_channels * 4, 1, bias=False),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden_channels * 4,
                nb_channels * self.nb_output_bins * 4,
                1,
                bias=False,
                groups=4,
            ),
            nn.BatchNorm1d(nb_channels * self.nb_output_bins * 4),
        )

    def forward(self, spec: torch.Tensor):
        batch, channels, bins, frames = spec.shape
        spec = spec[..., : self.max_bins, :]

        x = (
            spec.unsqueeze(1) + self.input_means.view(4, 1, -1, 1)
        ) * self.input_scale.view(4, 1, -1, 1)

        x = x.reshape(batch, -1, frames)
        cross_1 = self.affine1(x).view(batch, 4, -1, frames).mean(1)

        cross_1 = cross_1.permute(2, 0, 1)
        bass, *_ = self.bass_lstm(cross_1)
        drums, *_ = self.drums_lstm(cross_1)
        others, *_ = self.other_lstm(cross_1)
        vocals, *_ = self.vocals_lstm(cross_1)

        avg = (bass + drums + vocals + others) * 0.25
        cross_2 = torch.cat([cross_1, avg], 2).permute(1, 2, 0)

        mask = self.affine2(cross_2).view(
            batch, 4, channels, bins, frames
        ) * self.output_scale.view(4, 1, -1, 1) + self.output_means.view(4, 1, -1, 1)
        return mask.relu()
