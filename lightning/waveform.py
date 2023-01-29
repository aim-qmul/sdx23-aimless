import pytorch_lightning as pl
import torch
from torch import nn
from typing import List

from loss.time import TLoss, SDR
from augment.cuda import *

from utils import MDX_SOURCES, SDX_SOURCES


class WaveformSeparator(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 criterion: TLoss,
                 apply_transforms: bool = False,
                 use_sdx_targets: bool = False,
                 targets: List[str] = ['vocals', 'drums', 'bass', 'other'],
                 ):
        super().__init__()

        self.model = model
        self.criterion = criterion
        self.sdr = SDR()

        transforms = [
            RandomPitch(),
            SpeedPerturb(),
        ] if apply_transforms else []

        self.sources = SDX_SOURCES if use_sdx_targets else MDX_SOURCES

        self.transforms = nn.Sequential(*transforms)
        self.register_buffer('targets_idx', torch.tensor(
            sorted([self.sources.index(target) for target in targets])))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if len(self.transforms) > 0:
            y = self.transforms(y)
            x = y.sum(1)
        y = y[:, self.targets_idx].squeeze(1)

        pred = self.model(x)
        loss, values = self.criterion(pred, y, x)

        values['loss'] = loss
        self.log_dict(values, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, self.targets_idx].squeeze(1)

        pred = self.model(x)
        loss, values = self.criterion(pred, y, x)

        batch = pred.shape[0]
        sdrs = self.sdr(
            pred.view(-1, *pred.shape[-2:]), y.view(-1, *y.shape[-2:])).view(batch, -1).mean(0)

        for i, t in enumerate(self.targets_idx):
            values[f'{self.sources[t]}_sdr'] = sdrs[i].item()
        values['avg_sdr'] = sdrs.mean().item()
        return loss, values

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = sum(x[0] for x in outputs) / len(outputs)
        avg_values = {}
        for k in outputs[0][1].keys():
            avg_values[k] = sum(x[1][k] for x in outputs) / len(outputs)

        self.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.log_dict(avg_values, prog_bar=False, sync_dist=True)
