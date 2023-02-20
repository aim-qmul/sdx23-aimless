from typing import List, Optional, Callable
from pathlib import Path
import torchaudio
from tqdm import tqdm
from data.dataset import BaseDataset

from aimless.utils import SDX_SOURCES as SOURCES

__all__ = ["DnR", "source_idx"]


class DnR(BaseDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        seq_duration: float = 6.0,
        samples_per_track: int = 64,
        random: bool = False,
        random_track_mix: bool = False,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            root=root,
            split=split,
            seq_duration=seq_duration,
            samples_per_track=samples_per_track,
            random=random,
            random_track_mix=random_track_mix,
            transform=transform,
            sources=SOURCES,
            mix_name="mix",
        )

    def load_tracks(self, split: str):
        if self.split == "train":
            split_root = self.root / "tr"
        elif self.split == "valid":
            split_root = self.root / "cv"
        elif self.split == "test":
            split_root = self.root / "tt"
        else:
            raise ValueError("Invalid split: {}".format(self.split))

        tracks = sorted([x for x in split_root.iterdir() if x.is_dir()])
        for x in tracks:
            assert torchaudio.info(str(x / "mix.wav")).sample_rate == DnR.sr

        track_lenghts = [
            torchaudio.info(str(x / "mix.wav")).num_frames for x in tqdm(tracks)
        ]
        return tracks, track_lenghts
