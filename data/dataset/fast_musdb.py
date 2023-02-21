import torchaudio
from tqdm import tqdm
from data.dataset import BaseDataset
import musdb
import os
import yaml
from pathlib import Path
from typing import List

from aimless.utils import MDX_SOURCES as SOURCES

__all__ = ["FastMUSDB"]


class FastMUSDB(BaseDataset):
    def __init__(
        self,
        root: str,
        subsets: List[str] = ["train", "test"],
        split: str = None,
        **kwargs
    ):
        tracks, track_lengths = load_tracks(root, subsets, split)
        super().__init__(
            **kwargs,
            tracks=tracks,
            track_lengths=track_lengths,
            sources=SOURCES,
            mix_name="mixture",
        )


def load_tracks(root, subsets=None, split=None):
    root = Path(os.path.expanduser(root))
    setup_path = os.path.join(musdb.__path__[0], "configs", "mus.yaml")
    with open(setup_path, "r") as f:
        setup = yaml.safe_load(f)
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
        subset_folder = root / subset
        for _, folders, _ in tqdm(os.walk(subset_folder)):
            # parse pcm tracks and sort by name
            for track_name in sorted(folders):
                if subset == "train":
                    if split == "train" and track_name in setup["validation_tracks"]:
                        continue
                    elif (
                        split == "valid"
                        and track_name not in setup["validation_tracks"]
                    ):
                        continue

                track_folder = subset_folder / track_name
                # add track to list of tracks
                tracks.append(track_folder)
                meta = torchaudio.info(os.path.join(track_folder, "mixture.wav"))
                assert meta.sample_rate == FastMUSDB.sr

                track_lengths.append(meta.num_frames)

    return tracks, track_lengths
