import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTM(nn.Module):
    def __init__(self, emb_size: int, hidden_size: int, num_sources: int, **kwargs):
        super().__init__()
        self.embbeding = nn.Embedding(1024, emb_size)
        self.lstm = nn.LSTM(
            input_size=emb_size * 16,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
            **kwargs,
        )
        self.fc = nn.Linear(hidden_size * 2, num_sources * 16 * 1024)

    def forward(self, x):
        x = self.embbeding(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x, _ = self.lstm(x)
        x = self.fc(x).view(x.shape[0], x.shape[1], 1024, -1, 16).permute(0, 2, 3, 1, 4)
        return x
