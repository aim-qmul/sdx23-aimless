from torch.utils.data import Dataset
import torch
import random
import os
from pathlib import Path
import numpy as np
import torchaudio
from tqdm import tqdm
from typing import Optional, Callable


__all__ = ['DnR', 'source_idx']

SOURCES = ['music', 'sfx', 'speech']


def source_idx(source):
    return SOURCES.index(source)


class DnR(Dataset):
    sr: int = 44100

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 seq_duration: float = 6.0,
                 samples_per_track: int = 64,
                 random: bool = False,
                 random_track_mix: bool = False,
                 transform: Optional[Callable] = None
                 ):
        super().__init__()
        root = Path(os.path.expanduser(root))
        self.root = root
        self.seq_duration = seq_duration
        self.segment = int(self.seq_duration * self.sr)
        self.split = split
        self.samples_per_track = samples_per_track
        self.random_track_mix = random_track_mix
        self.random = random

        self.transform = transform

        if self.split == 'train':
            split_root = root / 'tr'
        elif self.split == 'valid':
            split_root = root / 'cv'
        elif self.split == 'test':
            split_root = root / 'tt'
        else:
            raise ValueError('Invalid split: {}'.format(self.split))

        self.tracks = sorted([x for x in split_root.iterdir() if x.is_dir()])
        self.track_lenghts = [torchaudio.info(
            str(x / 'mixture.wav')).num_frames for x in tqdm(self.tracks)]

        if self.seq_duration <= 0:
            self._size = len(self.tracks)
        elif self.random:
            self._size = len(self.tracks) * self.samples_per_track
        else:
            chunks = [l // self.segment for l in self.track_lenghts]
            cum_chunks = np.cumsum(chunks)
            self.cum_chunks = cum_chunks
            self._size = cum_chunks[-1]

    def __len__(self):
        return self._size

    def _get_random_track_idx(self):
        return random.randrange(len(self.tracks))

    def _get_random_start(self, length):
        return random.randrange(length - self.segment + 1)

    def _get_track_from_chunk(self, index):
        track_idx = np.digitize(index, self.cum_chunks)
        if track_idx > 0:
            chunk_start = (
                index - self.cum_chunks[track_idx - 1]) * self.segment
        else:
            chunk_start = index * self.segment
        return self.tracks[track_idx], chunk_start

    def __getitem__(self, index):
        stems = []
        if self.seq_duration <= 0:
            folder_name = self.tracks[index]
            x = torchaudio.load(
                folder_name / 'mix.wav',
            )[0]
            for s in SOURCES:
                source_name = folder_name / (s + '.wav')
                audio = torchaudio.load(source_name)[0]
                stems.append(audio)
        else:
            if self.random:
                track_idx = index // self.samples_per_track
                folder_name, chunk_start = self.tracks[track_idx], self._get_random_start(
                    self.track_lenghts[track_idx])
            else:
                folder_name, chunk_start = self._get_track_from_chunk(index)
            for s in SOURCES:
                if self.random_track_mix and self.random:
                    track_idx = self._get_random_track_idx()
                    folder_name, chunk_start = self.tracks[track_idx], self._get_random_start(
                        self.track_lenghts[track_idx])
                source_name = folder_name / (s + '.wav')
                audio = torchaudio.load(
                    source_name, num_frames=self.segment, frame_offset=chunk_start,
                )[0]
                stems.append(audio)
            if self.random_track_mix and self.random:
                x = sum(stems)
            else:
                x = torchaudio.load(
                    folder_name / 'mix.wav',
                    num_frames=self.segment, frame_offset=chunk_start,
                )[0]

        y = torch.stack(stems)
        if self.transform is not None:
            x, y = self.transform((x, y))
        return x, y
