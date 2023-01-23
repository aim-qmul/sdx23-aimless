from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import pytorch_lightning as pl


from data import FastMUSDB
from augment.cpu import *


class MUSDB(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        segment: int = 262144,
        samples_per_track: int = 64,
        random: bool = False,
        random_track_mix: bool = False,
        apply_transforms: bool = False,
        batch_size: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):

        if stage == "fit":
            transforms = (
                Compose([RandomGain(), RandomSwapLR(), RandomFlipPhase()])
                if self.hparams.apply_transforms
                else None
            )
            self.train_dataset = FastMUSDB(
                root=self.hparams.root,
                subsets=["train"],
                segment=self.hparams.segment,
                samples_per_track=self.hparams.samples_per_track,
                random=self.hparams.random,
                random_track_mix=self.hparams.random_track_mix,
                transform=transforms,
            )

        if stage == "validate" or stage == "fit":
            self.val_dataset = FastMUSDB(
                root=self.hparams.root,
                subsets=["test"],
                segment=self.hparams.segment,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )
