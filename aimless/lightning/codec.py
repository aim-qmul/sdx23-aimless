import pytorch_lightning as pl
import torch
from torch import nn
from typing import List, Dict

from ..loss.time import TLoss, SDR
from ..augment import CudaBase

from ..utils import MDX_SOURCES, SDX_SOURCES

from encodec import EncodecModel
from encodec.utils import convert_audio


class CodecSeparator(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        transforms: List[CudaBase] = None,
        use_sdx_targets: bool = False,
        targets: Dict[str, None] = {},
        codec_bandwidth: float = 24.0,
    ):
        super().__init__()

        # Instantiate a pretrained EnCodec model
        self.codec = EncodecModel.encodec_model_48khz()
        self.codec.set_target_bandwidth(codec_bandwidth)

        self.model = model
        self.criterion = criterion
        self.sdr = SDR()

        if transforms is None:
            transforms = []

        self.sources = SDX_SOURCES if use_sdx_targets else MDX_SOURCES

        self.transforms = nn.Sequential(*transforms)
        self.register_buffer(
            "targets_idx",
            torch.tensor(sorted([self.sources.index(target) for target in targets])),
        )

    def separate(self, x: torch.Tensor):
        """Given a mixture waveform, convert to codes,
        predict stem codes, and then decode back to stem waveforms.

        """
        mixture_codes = self.codec.encode(x)
        stem_codes = self.model(mixture_codes)
        stems = self.codec.decode(stem_codes)
        return stems

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # convert mixture waveform to codes
        mixture_codes = self.codec.encode(y)

        # convert stems to codes for loss
        stem_codes = self.codec.encode(x)

        pred_stem_codes = self.model(x)

        self.log("loss", prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, self.targets_idx].squeeze(1)

        pred = self.model(x)
        loss, values = self.criterion(pred, y, x)

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
