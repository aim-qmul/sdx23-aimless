import pytorch_lightning as pl
import torch
from torch import nn
from typing import List


from utils import MDX_SOURCES, SDX_SOURCES


class CodeSeparator(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        targets: List[str] = ["vocals", "drums", "bass", "other"],
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("train-loss", loss, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("val-loss", loss, prog_bar=False, sync_dist=True)
        return loss
