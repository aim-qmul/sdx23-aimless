import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Dict
from encodec import EncodecModel
from torchaudio.transforms import Resample

from ..loss.time import SDR
from ..augment import CudaBase

from ..utils import MDX_SOURCES, SDX_SOURCES


class SymbolicSeparator(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        transforms: List[CudaBase] = None,
        use_sdx_targets: bool = False,
        targets: Dict[str, None] = {},
    ):
        super().__init__()

        self.model = model
        self.sdr = SDR()
        self.criterion = nn.CrossEntropyLoss()

        if transforms is None:
            transforms = []

        self.sources = SDX_SOURCES if use_sdx_targets else MDX_SOURCES

        self.transforms = nn.Sequential(*transforms)
        self.register_buffer(
            "targets_idx",
            torch.tensor(sorted([self.sources.index(target) for target in targets])),
        )

        self.encodec = EncodecModel.encodec_model_48khz()
        self.encodec.set_target_bandwidth(3)
        for p in self.encodec.parameters():
            p.requires_grad = False

        self.resampler = Resample(44100, 48000)

    def get_codes(self, x):
        with torch.no_grad():
            emb = self.encodec.encoder(x)
            codes = self.encodec.quantizer.encode(
                emb, self.encodec.frame_rate, self.encodec.bandwidth
            )
        return codes

    def decode(self, codes):
        with torch.no_grad():
            emb = self.encodec.quantizer.decode(codes)
            x = self.encodec.decoder(emb)
        return x

    def convert2stereo48k(self, x):
        x = self.resampler(x)
        if x.shape[-2] == 1:
            x = torch.cat([x, x], dim=-2)
        return x

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if len(self.transforms) > 0:
            y = self.transforms(y)
            x = y.sum(1)
        y = y[:, self.targets_idx]

        # convert to stereo 48k
        x = self.convert2stereo48k(x)
        y = self.convert2stereo48k(y)

        z = self.get_codes(x).permute(1, 2, 0)
        targets = self.get_codes(y.view(-1, *y.shape[-2:]))
        targets = targets.view(
            16, -1, len(self.targets_idx), targets.shape[-1]
        ).permute(1, 2, 3, 0)

        pred = self.model(z)
        loss = self.criterion(pred, targets)

        self.log("loss", loss, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, self.targets_idx]
        N = y.shape[0]

        # convert to stereo 48k
        x = self.convert2stereo48k(x)
        y = self.convert2stereo48k(y)

        z = self.get_codes(x).permute(1, 2, 0)
        targets = self.get_codes(y.view(-1, *y.shape[-2:]))
        targets = targets.view(
            16, -1, len(self.targets_idx), targets.shape[-1]
        ).permute(1, 2, 3, 0)

        pred = self.model(z)
        loss = self.criterion(pred, targets)

        pred_codes = pred.argmax(1).permute(3, 0, 1, 2)
        pred_codes = pred_codes.reshape(16, -1, pred_codes.shape[-1])
        pred = self.decode(pred_codes)
        pred = pred.view(N, len(self.targets_idx), *pred.shape[-2:])

        sdrs = (
            self.sdr(pred.view(-1, *pred.shape[-2:]), y.view(-1, *y.shape[-2:]))
            .view(N, -1)
            .mean(0)
        )

        values = {}
        for i, t in enumerate(self.targets_idx):
            values[f"{self.sources[t]}_sdr"] = sdrs[i].item()
        values["avg_sdr"] = sdrs.mean().item()
        return loss, values

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = sum(x[0] for x in outputs) / len(outputs)
        avg_values = {}
        for k in outputs[0][1].keys():
            avg_values[k] = sum(x[1][k] for x in outputs) / len(outputs)

        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log_dict(avg_values, prog_bar=False, sync_dist=True)


class ARSeparator(SymbolicSeparator):
    def training_step(self, batch, batch_idx):
        x, y = batch
        if len(self.transforms) > 0:
            y = self.transforms(y)
            x = y.sum(1)
        y = y[:, self.targets_idx]
        N = y.shape[0]

        # convert to stereo 48k
        x = self.convert2stereo48k(x)
        y = self.convert2stereo48k(y)

        # # get std and mean
        # x_std = x.std((1, 2))
        # x = x / x_std[:, None, None]
        # y = y / x_std[:, None, None, None]

        x_codes = self.get_codes(x).permute(1, 2, 0)
        y_codes = (
            self.get_codes(y.view(-1, *y.shape[-2:]))
            .view(2, N, len(self.targets_idx), -1)
            .permute(1, 2, 3, 0)
        )
        y_past = torch.cat(
            [torch.zeros_like(y_codes[:, :, :1]), y_codes[:, :, :-1]], dim=2
        )

        # create causal mask
        T = y_past.shape[2]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=self.device
        )

        pred = self.model(x_codes, y_past, tgt_mask=causal_mask)
        loss = self.criterion(pred, y_codes)

        self.log("loss", loss, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, self.targets_idx]
        N = y.shape[0]

        # convert to stereo 48k
        x = self.convert2stereo48k(x)
        y = self.convert2stereo48k(y)

        # # get std and mean
        # x_std = x.std((1, 2))
        # x = x / x_std[:, None, None]
        # y_rescaled = y / x_std[:, None, None, None]

        x_codes = self.get_codes(x).permute(1, 2, 0)
        y_codes = (
            self.get_codes(y.view(-1, *y.shape[-2:]))
            .view(2, N, len(self.targets_idx), -1)
            .permute(1, 2, 3, 0)
        )
        y_past = torch.cat(
            [torch.zeros_like(y_codes[:, :, :1]), y_codes[:, :, :-1]], dim=2
        )

        # create causal mask
        T = y_past.shape[2]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=self.device
        )

        pred = self.model(x_codes, y_past, tgt_mask=causal_mask)
        loss = self.criterion(pred, y_codes)

        pred_codes = pred.argmax(1).permute(3, 0, 1, 2)
        pred_codes = pred_codes.reshape(2, -1, pred_codes.shape[-1])
        pred = self.decode(pred_codes)
        pred = pred.view(N, len(self.targets_idx), *pred.shape[-2:])

        # scale back
        # pred = pred * x_std[:, None, None, None]

        sdrs = (
            self.sdr(pred.view(-1, *pred.shape[-2:]), y.view(-1, *y.shape[-2:]))
            .view(N, -1)
            .mean(0)
        )

        values = {}
        for i, t in enumerate(self.targets_idx):
            values[f"{self.sources[t]}_sdr"] = sdrs[i].item()
        values["avg_sdr"] = sdrs.mean().item()
        return loss, values
