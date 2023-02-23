import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS


class PretrainedHDemucs(torch.nn.Module):
    def __init__(self, sources, download_weights: bool = False) -> None:
        super().__init__()
        if download_weights:
            bundle = HDEMUCS_HIGH_MUSDB_PLUS
            self.model = bundle.get_model()
        else:
            self.model = torchaudio.models.hdemucs_high()

    def forward(self, x: torch.Tensor):
        return self.model(x)
