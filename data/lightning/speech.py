from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import List, Optional
import torch

from data.dataset.speech import SpeechNoise as SpeechNoiseDataset
from data.augment import CPUBase


class SpeechNoise(pl.LightningDataModule):
    def __init__(
        self,
        speech_root: str,
        noise_root: str,
        seq_duration: float = 6.0,
        samples_per_track: int = 64,
        least_overlap_ratio: float = 0.5,
        snr_sampler: Optional[torch.distributions.Distribution] = None,
        mono: bool = True,
        transforms: List[CPUBase] = None,
        batch_size: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters(
            "speech_root",
            "noise_root",
            "seq_duration",
            "samples_per_track",
            "least_overlap_ratio",
            "snr_sampler",
            "mono",
            "batch_size",
        )
        # manually save transforms since pedalboard is not pickleable
        if transforms is None:
            self.transforms = None
        else:
            self.transforms = Compose(transforms)

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = SpeechNoiseDataset(
                speech_root=self.hparams.speech_root,
                noise_root=self.hparams.noise_root,
                seq_duration=self.hparams.seq_duration,
                samples_per_track=self.hparams.samples_per_track,
                least_overlap_ratio=self.hparams.least_overlap_ratio,
                snr_sampler=self.hparams.snr_sampler,
                mono=self.hparams.mono,
                transform=self.transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )
