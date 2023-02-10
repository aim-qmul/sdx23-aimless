import musdb
from torch.utils.data import Dataset
import torch
import random
import os
import yaml
import numpy as np
import torchaudio
from tqdm import tqdm
from typing import Optional, Callable

from utils import MDX_SOURCES as SOURCES


__all__ = ["FastMUSDB", "source_idx"]


class FastMUSDB(Dataset):
    sr: int = 44100

    def __init__(
        self,
        root=None,
        subsets=["train", "test"],
        split=None,
        seq_duration=6.0,
        samples_per_track=64,
        random=False,
        random_track_mix=False,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.seq_duration = seq_duration
        self.subsets = subsets
        self.segment = int(self.seq_duration * self.sr)
        self.split = split
        self.samples_per_track = samples_per_track
        self.random_track_mix = random_track_mix
        self.random = random

        self.transform = transform

        setup_path = os.path.join(musdb.__path__[0], "configs", "mus.yaml")
        with open(setup_path, "r") as f:
            self.setup = yaml.safe_load(f)

        self.tracks, self.track_lenghts = self.load_mus_tracks(
            self.sr, self.subsets, self.split
        )

        if self.seq_duration <= 0:
            self._size = len(self.tracks)
        elif self.random:
            self._size = len(self.tracks) * self.samples_per_track
        else:
            chunks = [l // self.segment for l in self.track_lenghts]
            cum_chunks = np.cumsum(chunks)
            self.cum_chunks = cum_chunks
            self._size = cum_chunks[-1]

    def load_mus_tracks(self, sr, subsets=None, split=None):
        if subsets is not None:
            if isinstance(subsets, str):
                subsets = [subsets]
        else:
            subsets = ["train", "test"]

        if subsets != ["train"] and split is not None:
            raise RuntimeError("Subset has to set to `train` when split is used")

        print("Gathering files ...")
        tracks = []
        track_lengths = []
        for subset in subsets:
            subset_folder = os.path.join(self.root, subset)
            for _, folders, _ in tqdm(os.walk(subset_folder)):
                # parse pcm tracks and sort by name
                for track_name in sorted(folders):
                    if subset == "train":
                        if (
                            split == "train"
                            and track_name in self.setup["validation_tracks"]
                        ):
                            continue
                        elif (
                            split == "valid"
                            and track_name not in self.setup["validation_tracks"]
                        ):
                            continue

                    track_folder = os.path.join(subset_folder, track_name)
                    # add track to list of tracks
                    tracks.append(track_folder)
                    meta = torchaudio.info(os.path.join(track_folder, "mixture.wav"))
                    assert meta.sample_rate == sr

                    track_lengths.append(meta.num_frames)

        return tracks, track_lengths

    def __len__(self):
        return self._size

    def _get_random_track_idx(self):
        return random.randrange(len(self.tracks))

    def _get_random_start(self, length):
        return random.randrange(length - self.segment + 1)

    def _get_track_from_chunk(self, index):
        track_idx = np.digitize(index, self.cum_chunks)
        if track_idx > 0:
            chunk_start = (index - self.cum_chunks[track_idx - 1]) * self.segment
        else:
            chunk_start = index * self.segment
        return self.tracks[track_idx], chunk_start

    def __getitem__(self, index):
        stems = []
        if self.seq_duration <= 0:
            folder_name = self.tracks[index]
            x = torchaudio.load(
                os.path.join(folder_name, "mixture.wav"),
            )[0]
            for s in SOURCES:
                source_name = os.path.join(folder_name, s + ".wav")
                audio = torchaudio.load(source_name)[0]
                stems.append(audio)
        else:
            if self.random:
                track_idx = index // self.samples_per_track
                folder_name, chunk_start = self.tracks[
                    track_idx
                ], self._get_random_start(self.track_lenghts[track_idx])
            else:
                folder_name, chunk_start = self._get_track_from_chunk(index)
            for s in SOURCES:
                if self.random_track_mix and self.random:
                    track_idx = self._get_random_track_idx()
                    folder_name, chunk_start = self.tracks[
                        track_idx
                    ], self._get_random_start(self.track_lenghts[track_idx])
                source_name = os.path.join(folder_name, s + ".wav")
                audio = torchaudio.load(
                    source_name,
                    num_frames=self.segment,
                    frame_offset=chunk_start,
                )[0]
                stems.append(audio)
            if self.random_track_mix and self.random:
                x = sum(stems)
            else:
                x = torchaudio.load(
                    os.path.join(folder_name, "mixture.wav"),
                    num_frames=self.segment,
                    frame_offset=chunk_start,
                )[0]

        y = torch.stack(stems)

        if self.transform is not None:
            x, y = self.transform((x, y))

        return x, y
