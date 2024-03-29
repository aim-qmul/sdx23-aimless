from torch.utils.data import Dataset
import torch
import random
import numpy as np
import torchaudio
from typing import Optional, Callable, List, Tuple
from pathlib import Path

__all__ = ["BaseDataset"]


class BaseDataset(Dataset):
    sr: int = 44100

    def __init__(
        self,
        tracks: List[Path],
        track_lengths: List[int],
        sources: List[str],
        mix_name: str = "mix",
        seq_duration: float = 6.0,
        samples_per_track: int = 64,
        random: bool = False,
        random_track_mix: bool = False,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.tracks = tracks
        self.track_lengths = track_lengths
        self.sources = sources
        self.mix_name = mix_name
        self.seq_duration = seq_duration
        self.samples_per_track = samples_per_track
        self.segment = int(self.seq_duration * self.sr)
        self.random = random
        self.random_track_mix = random_track_mix
        self.transform = transform

        if self.seq_duration <= 0:
            self._size = len(self.tracks)
        elif self.random:
            self._size = len(self.tracks) * self.samples_per_track
        else:
            chunks = [l // self.segment for l in self.track_lengths]
            cum_chunks = np.cumsum(chunks)
            self.cum_chunks = cum_chunks
            self._size = cum_chunks[-1]

    def load_tracks(self) -> Tuple[List[Path], List[int]]:
        # Implement in child class
        # Return list of tracks and list of track lengths
        raise NotImplementedError

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
                folder_name / f"{self.mix_name}.wav",
            )[0]
            for s in self.sources:
                source_name = folder_name / (s + ".wav")
                audio = torchaudio.load(source_name)[0]
                stems.append(audio)
        else:
            if self.random:
                track_idx = index // self.samples_per_track
                folder_name, chunk_start = self.tracks[
                    track_idx
                ], self._get_random_start(self.track_lengths[track_idx])
            else:
                folder_name, chunk_start = self._get_track_from_chunk(index)
            for s in self.sources:
                if self.random_track_mix and self.random:
                    track_idx = self._get_random_track_idx()
                    folder_name, chunk_start = self.tracks[
                        track_idx
                    ], self._get_random_start(self.track_lengths[track_idx])
                source_name = folder_name / (s + ".wav")
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
                    folder_name / f"{self.mix_name}.wav",
                    num_frames=self.segment,
                    frame_offset=chunk_start,
                )[0]
        y = torch.stack(stems)
        if self.transform is not None:
            x, y = self.transform((x, y))
        return x, y
