from torchvision.transforms import Compose
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl


from data import FastMUSDB
from data import DnR as DnRDataset
from augment.cpu import *


class MUSDB(pl.LightningDataModule):
    def __init__(self,
                 root: str,
                 seq_duration: float = 6.0,
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
            transforms = Compose([
                RandomGain(),
                RandomSwapLR(),
                RandomFlipPhase(),
                LimitAug(sample_rate=44100)
            ]) if self.hparams.apply_transforms else None
            self.train_dataset = FastMUSDB(root=self.hparams.root,
                                           subsets=['train'],
                                           seq_duration=self.hparams.seq_duration,
                                           samples_per_track=self.hparams.samples_per_track,
                                           random=self.hparams.random,
                                           random_track_mix=self.hparams.random_track_mix,
                                           transform=transforms)

        if stage == "validate" or stage == "fit":
            self.val_dataset = FastMUSDB(root=self.hparams.root,
                                         subsets=['test'], seq_duration=self.hparams.seq_duration)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,  num_workers=4, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=4)


class DnR(pl.LightningDataModule):
    def __init__(self,
                 root: str,
                 seq_duration: float = 6.0,
                 samples_per_track: int = 64,
                 random: bool = False,
                 include_val: bool = False,
                 random_track_mix: bool = False,
                 apply_transforms: bool = False,
                 batch_size: int = 16,
                 ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):

        if stage == "fit":
            transforms = Compose([
                RandomGain(),
                RandomFlipPhase(),
                LimitAug(sample_rate=44100)
            ]) if self.hparams.apply_transforms else None
            self.train_dataset = DnRDataset(root=self.hparams.root,
                                            split='train',
                                            seq_duration=self.hparams.seq_duration,
                                            samples_per_track=self.hparams.samples_per_track,
                                            random=self.hparams.random,
                                            random_track_mix=self.hparams.random_track_mix,
                                            transform=transforms)

            if self.hparams.include_val:
                self.train_dataset = ConcatDataset([
                    self.train_dataset,
                    DnRDataset(root=self.hparams.root,
                               split='valid',
                               seq_duration=self.hparams.seq_duration,
                               samples_per_track=self.hparams.samples_per_track,
                               random=self.hparams.random,
                               random_track_mix=self.hparams.random_track_mix,
                               transform=transforms)
                ])

        if stage == "validate" or stage == "fit":
            self.val_dataset = DnRDataset(root=self.hparams.root,
                                          split='valid',
                                          seq_duration=self.hparams.seq_duration)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,  num_workers=4, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=4)
