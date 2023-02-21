import torchaudio
from tqdm import tqdm
from data.dataset import BaseDataset
from pathlib import Path
import os

from aimless.utils import SDX_SOURCES as SOURCES

__all__ = ["DnR"]


class DnR(BaseDataset):
    def __init__(self, root: str, split: str, **kwargs):
        tracks, track_lengths = load_tracks(root, split)
        super().__init__(
            **kwargs,
            tracks=tracks,
            track_lengths=track_lengths,
            sources=SOURCES,
            mix_name="mix",
        )


def load_tracks(root: str, split: str):
    root = Path(os.path.expanduser(root))
    if split == "train":
        split_root = root / "tr"
    elif split == "valid":
        split_root = root / "cv"
    elif split == "test":
        split_root = root / "tt"
    else:
        raise ValueError("Invalid split: {}".format(split))
    tracks = sorted([x for x in split_root.iterdir() if x.is_dir()])
    for x in tracks:
        assert torchaudio.info(str(x / "mix.wav")).sample_rate == DnR.sr

    track_lengths = [
        torchaudio.info(str(x / "mix.wav")).num_frames for x in tqdm(tracks)
    ]
    return tracks, track_lengths
