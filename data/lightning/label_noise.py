from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import List
import os

from data.dataset import LabelNoiseBleed
from data.augment import CPUBase


class LabelNoise(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        seq_duration: float = 6.0,
        samples_per_track: int = 64,
        random: bool = False,
        random_track_mix: bool = False,
        transforms: List[CPUBase] = None,
        batch_size: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters(
            "root",
            "seq_duration",
            "samples_per_track",
            "random",
            "random_track_mix",
            "batch_size",
        )
        # manually save transforms since pedalboard is not pickleable
        if transforms is None:
            self.transforms = None
        else:
            self.transforms = Compose(transforms)

    def setup(self, stage=None):
        label_noise_path = None
        current_dir = os.path.dirname(os.path.realpath(__file__))
        if "label_noise.csv" in os.listdir(current_dir):
            label_noise_path = os.path.join(current_dir, "label_noise.csv")
        assert label_noise_path is not None
        if stage == "fit":
            self.train_dataset = LabelNoiseBleed(
                root=self.hparams.root,
                split="train",
                seq_duration=self.hparams.seq_duration,
                samples_per_track=self.hparams.samples_per_track,
                random=self.hparams.random,
                random_track_mix=self.hparams.random_track_mix,
                transform=self.transforms,
                clean_csv_path=label_noise_path,
            )

        if stage == "validate" or stage == "fit":
            self.val_dataset = LabelNoiseBleed(
                root=self.hparams.root,
                split="train",
                seq_duration=self.hparams.seq_duration,
                random=self.hparams.random,
                random_track_mix=self.hparams.random_track_mix,
                transform=self.transforms,
                clean_csv_path=label_noise_path,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=True,
        )
