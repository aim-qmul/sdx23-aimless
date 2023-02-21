import torchaudio
from tqdm import tqdm
from data.dataset import BaseDataset

from aimless.utils import MDX_SOURCES as SOURCES

__all__ = ["DnR"]


# Run scripts/dataset_split_and_mix.py first!
class LabelNoiseBleed(BaseDataset):
    def __init__(self, split: str = "train", **kwargs):
        super().__init__(
            **kwargs,
            split=split,
            sources=SOURCES,
            mix_name="mixture",
        )

    def load_tracks(self):
        if self.split == "train":
            split_root = self.root / "train"
        elif self.split == "valid":
            split_root = self.root / "valid"
        elif self.split == "test":
            split_root = self.root / "test"
        else:
            raise ValueError("Invalid split: {}".format(self.split))

        tracks = sorted([x for x in split_root.iterdir() if x.is_dir()])
        for x in tracks:
            assert (
                torchaudio.info(str(x / "mixture.wav")).sample_rate
                == LabelNoiseBleed.sr
            )

        track_lenghts = [
            torchaudio.info(str(x / "mixture.wav")).num_frames for x in tqdm(tracks)
        ]
        return tracks, track_lenghts
