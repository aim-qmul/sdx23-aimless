import torchaudio
from tqdm import tqdm
from data.dataset import BaseDataset
import musdb
import os
import yaml

from aimless.utils import MDX_SOURCES as SOURCES

__all__ = ["FastMUSDB"]


class FastMUSDB(BaseDataset):
    def __init__(self, subsets=["train", "test"], **kwargs):
        self.subsets = subsets
        super().__init__(
            **kwargs,
            sources=SOURCES,
            mix_name="mixture",
        )

    def load_tracks(self):
        setup_path = os.path.join(musdb.__path__[0], "configs", "mus.yaml")
        with open(setup_path, "r") as f:
            self.setup = yaml.safe_load(f)
        if self.subsets is not None:
            if isinstance(self.subsets, str):
                self.subsets = [self.subsets]
        else:
            self.subsets = ["train", "test"]

        if self.subsets != ["train"] and self.split is not None:
            raise RuntimeError("Subset has to set to `train` when split is used")

        print("Gathering files ...")
        tracks = []
        track_lengths = []
        for subset in self.subsets:
            subset_folder = self.root / subset
            for _, folders, _ in tqdm(os.walk(subset_folder)):
                # parse pcm tracks and sort by name
                for track_name in sorted(folders):
                    if subset == "train":
                        if (
                            self.split == "train"
                            and track_name in self.setup["validation_tracks"]
                        ):
                            continue
                        elif (
                            self.split == "valid"
                            and track_name not in self.setup["validation_tracks"]
                        ):
                            continue

                    track_folder = subset_folder / track_name
                    # add track to list of tracks
                    tracks.append(track_folder)
                    meta = torchaudio.info(os.path.join(track_folder, "mixture.wav"))
                    assert meta.sample_rate == FastMUSDB.sr

                    track_lengths.append(meta.num_frames)

        return tracks, track_lengths
