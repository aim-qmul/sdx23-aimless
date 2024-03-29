import pytorch_lightning as pl
import torch
from torch import nn
from typing import List, Dict
from torchaudio.transforms import Spectrogram, InverseSpectrogram

from ..loss.time import SDR
from ..loss.freq import FLoss
from ..augment import CudaBase

from ..utils import MWF, MDX_SOURCES, SDX_SOURCES, SE_SOURCES


class MaskPredictor(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: FLoss,
        transforms: List[CudaBase] = None,
        target_track: str = None,
        targets: Dict[str, None] = {},
        n_fft: int = 4096,
        hop_length: int = 1024,
        **mwf_kwargs,
    ):
        super().__init__()

        self.model = model
        self.criterion = criterion
        self.sdr = SDR()
        self.mwf = MWF(**mwf_kwargs)
        self.spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
        self.inv_spec = InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)

        if transforms is None:
            transforms = []

        self.transforms = nn.Sequential(*transforms)
        if target_track == "sdx":
            self.sources = SDX_SOURCES
        elif target_track == "mdx":
            self.sources = MDX_SOURCES
        elif target_track == "se":
            self.sources = SE_SOURCES
        else:
            raise ValueError(f"Invalid target track: {target_track}")
        self.register_buffer(
            "targets_idx",
            torch.tensor(sorted([self.sources.index(target) for target in targets])),
        )

    def forward(self, x):
        X = self.spec(x)
        X_mag = X.abs()
        pred_mask = self.model(X_mag)
        Y = self.mwf(pred_mask, X)
        pred = self.inv_spec(Y)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        if len(self.transforms) > 0:
            y = self.transforms(y)
            x = y.sum(1)
        y = y[:, self.targets_idx].squeeze(1)

        X = self.spec(x)
        Y = self.spec(y)
        X_mag = X.abs()
        pred_mask = self.model(X_mag)
        loss, values = self.criterion(pred_mask, Y, X, y, x)

        values["loss"] = loss
        self.log_dict(values, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, self.targets_idx].squeeze(1)

        X = self.spec(x)
        Y = self.spec(y)
        X_mag = X.abs()
        pred_mask = self.model(X_mag)
        loss, values = self.criterion(pred_mask, Y, X, y, x)

        pred = self.inv_spec(self.mwf(pred_mask, X))

        batch = pred.shape[0]
        sdrs = (
            self.sdr(pred.view(-1, *pred.shape[-2:]), y.view(-1, *y.shape[-2:]))
            .view(batch, -1)
            .mean(0)
        )

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
