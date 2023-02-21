import torchaudio
from tqdm import tqdm
from data.dataset import BaseDataset
import os
from pathlib import Path

from aimless.utils import MDX_SOURCES as SOURCES

__all__ = ["DnR"]


# Run scripts/dataset_split_and_mix.py first!
class LabelNoiseBleed(BaseDataset):
    def __init__(self, root: str, split: str, **kwargs):
        tracks, track_lengths = load_tracks(root, split)
        super().__init__(
            **kwargs,
            tracks=tracks,
            track_lengths=track_lengths,
            sources=SOURCES,
            mix_name="mixture",
        )


def load_tracks(root: str, split: str):
    root = Path(os.path.expanduser(root))
    if split == "train":
        split_root = root / "train"
    elif split == "valid":
        split_root = root / "valid"
    elif split == "test":
        split_root = root / "test"
    else:
        raise ValueError("Invalid split: {}".format(split))

    tracks = sorted([x for x in split_root.iterdir() if x.is_dir()])
    for x in tracks:
        assert torchaudio.info(str(x / "mixture.wav")).sample_rate == LabelNoiseBleed.sr

    track_lengths = [
        torchaudio.info(str(x / "mixture.wav")).num_frames for x in tqdm(tracks)
    ]
    return tracks, track_lengths
