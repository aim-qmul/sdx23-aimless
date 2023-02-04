import os
import glob
import torch
import numpy as np

from utils import MDX_SOURCES as SOURCES


class CodedMUSDB(torch.utils.data.Dataset):
    def __init__(self, root: str, seq_len: int = 1024) -> None:
        super().__init__()
        self.root = root
        self.seq_len = seq_len
        # find all song directories
        song_dirs = glob.glob(os.path.join(root, "*"))
        song_dirs = [song_dir for song_dir in song_dirs if os.path.isdir(song_dir)]
        self.song_dirs = song_dirs * 100

    def _codes_to_tensor(self, encoded_frames):
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        # [B, n_q, T]
        return codes

    def __len__(self):
        return len(self.song_dirs)

    def __getitem__(self, idx):
        song_dir = self.song_dirs[idx]

        # get mixture code
        filepath = os.path.join(song_dir, "mixture.pt")
        x = self._codes_to_tensor(torch.load(filepath, map_location="cpu"))

        # get source codes
        sources = []
        for source in SOURCES:
            filepath = os.path.join(song_dir, f"{source}.pt")
            sources.append(
                self._codes_to_tensor(torch.load(filepath, map_location="cpu"))
            )
        y = torch.cat(sources, dim=0)

        # random time index
        start_idx = np.random.randint(0, y.shape[-1] - self.seq_len - 1)
        end_idx = start_idx + self.seq_len
        y = y[..., start_idx:end_idx].squeeze(0).float()
        x = x[..., start_idx:end_idx].squeeze(0).float()

        return x, y
