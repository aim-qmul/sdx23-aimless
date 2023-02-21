import torchaudio
from tqdm import tqdm
from data.dataset import BaseDataset

from aimless.utils import SDX_SOURCES as SOURCES

__all__ = ["DnR"]


class DnR(BaseDataset):
    def __init__(
        self,
        split: str = "train",
        **kwargs,
    ):
        super().__init__(
            **kwargs,
            split=split,
            sources=SOURCES,
            mix_name="mix",
        )

    def load_tracks(self):
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
