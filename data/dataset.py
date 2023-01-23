import torch

class SourceSeparationDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        return
    
    def __getitem__(self, idx:int):
        return 