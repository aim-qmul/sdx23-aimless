import torch
import torch.nn as nn
from torchaudio.transforms import Resample


@torch.jit.script
def glu(a, b):
    return a * b.sigmoid()


@torch.jit.script
def standardize(x, mu, std):
    return (x - mu) / std


@torch.jit.script
def destandardize(x, mu, std):
    return x * std + mu


def rescale_conv(reference):
    @torch.no_grad()
    def closure(m: nn.Module):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            std = m.weight.std()
            scale = (std / reference) ** 0.5
            m.weight.div_(scale)
            if m.bias is not None:
                m.bias.div_(scale)

    return closure


class DemucsSplit(nn.Module):
    def __init__(
        self,
        channels=64,
        depth=6,
        rescale=0.1,
        resample=True,
        kernel_size=8,
        stride=4,
        lstm_layers=2,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.depth = depth
        self.channels = channels

        if resample:
            self.up_sample = Resample(1, 2)
            self.down_sample = Resample(2, 1)

        self.encoder = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.dec_pre_convs = nn.ModuleList()
        self.decoder = nn.ModuleList()

        in_channels = 2
        for index in range(depth):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, channels, kernel_size, stride),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(channels, channels * 2, 1),
                    nn.GLU(dim=1),
                )
            )

            decode = []
            if index > 0:
                out_channels = in_channels * 2
            else:
                out_channels = 8

            self.convs_1x1.insert(
                0, nn.Conv1d(channels, channels * 4, 3, padding=1, bias=False)
            )
            self.dec_pre_convs.insert(
                0, nn.Conv1d(channels * 2, channels * 4, 3, padding=1, groups=4)
            )
            decode = [
                nn.ConvTranspose1d(
                    channels * 2, out_channels, kernel_size, stride, groups=4
                )
            ]
            if index > 0:
                decode.append(nn.ReLU(inplace=True))
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels *= 2

        channels = in_channels

        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=channels,
            num_layers=lstm_layers,
            dropout=0,
            bidirectional=True,
        )
        self.lstm_linear = nn.Linear(channels * 2, channels * 2)

        self.apply(rescale_conv(reference=rescale))

    def forward(self, x):
        length = x.size(2)

        mono = x.mean(1, keepdim=True)
        mu = mono.mean(dim=-1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True).add_(1e-5)
        x = standardize(x, mu, std)

        if hasattr(self, "up_sample"):
            x = self.up_sample(x)

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = self.lstm_linear(x).permute(1, 2, 0)

        for decode, pre_dec, conv1x1 in zip(
            self.decoder, self.dec_pre_convs, self.convs_1x1
        ):
            skip = saved.pop()

            x = pre_dec(x) + conv1x1(skip[..., : x.shape[2]])
            a, b = x.view(x.shape[0], 4, -1, x.shape[2]).chunk(2, 2)
            x = glu(a, b)
            x = decode(x.view(x.shape[0], -1, x.shape[3]))

        if hasattr(self, "down_sample"):
            x = self.down_sample(x)

        x = destandardize(x, mu, std)
        x = x.view(-1, 4, 2, x.size(-1))
        return x
