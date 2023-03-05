import torch
import torch.nn as nn
import torch.nn.functional as F

from .rel_tdemucs import PositionalEmbedding


class EncodecTransformer(nn.Module):
    def __init__(self, emb_dim: int, d_model: int, num_sources: int, **kwargs):
        super().__init__()
        self.embbeding = nn.Embedding(1024, emb_dim)
        self.linear1 = nn.Linear(emb_dim * 2, d_model, bias=False)
        self.linear2 = nn.Linear(emb_dim * 2 * num_sources, d_model, bias=False)
        self.model = nn.Transformer(
            d_model=d_model,
            batch_first=True,
            **kwargs,
        )
        self.linear3 = nn.Linear(d_model, num_sources * 2 * 1024)
        self.pos_emb = PositionalEmbedding(emb_dim)
        self.pos_proj1 = nn.Linear(emb_dim, d_model, bias=False)
        self.pos_proj2 = nn.Linear(emb_dim, d_model, bias=False)

    def forward(self, src, tgt, **kwargs):
        src = self.embbeding(src)
        src = src.view(src.shape[0], src.shape[1], -1)
        src = self.linear1(src)
        
        tgt = self.embbeding(tgt)
        tgt = tgt.permute(0, 2, 1, 3, 4).reshape(tgt.shape[0], tgt.shape[2], -1)
        tgt = self.linear2(tgt)

        T = src.shape[1]
        seq = torch.arange(T, device=src.device, dtype=src.dtype)
        pos_emb = self.pos_emb(seq)
        src = src + self.pos_proj1(pos_emb)
        tgt = tgt + self.pos_proj2(pos_emb)

        x = self.model(src, tgt, **kwargs)

        x = (
            self.linear3(x)
            .view(x.shape[0], x.shape[1], 1024, -1, 2)
            .permute(0, 2, 3, 1, 4)
        )
        return x
