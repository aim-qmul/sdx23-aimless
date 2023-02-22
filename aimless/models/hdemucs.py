import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS


class PretrainedHDemucs(torch.nn.Module):
    def __init__(self, sources) -> None:
        super().__init__()
        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        self.model = bundle.get_model()

    def forward(self, x: torch.Tensor):
        return self.model(x)
