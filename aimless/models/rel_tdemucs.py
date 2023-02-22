from math import inf
from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchaudio.transforms import Resample

from .demucs_split import standardize, destandardize, rescale_conv


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = pos_seq[:, None] * self.inv_freq
        pos_emb = torch.view_as_real(torch.exp(1j * sinusoid_inp)).view(
            pos_seq.size(0), -1
        )
        return pos_emb


class RelMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=True,
            **kwargs
        )
        self.register_parameter(
            "u", nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        )
        self.register_parameter(
            "v", nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        )
        self.pos_emb = PositionalEmbedding(self.embed_dim)
        self.pos_emb_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(self, x, mask=None):
        # x: [B, T, C]
        B, T, _ = x.size()
        seq = torch.arange(-T + 1, T, device=x.device)
        pos_emb = self.pos_emb_proj(self.pos_emb(seq))
        pos_emb = pos_emb.view(-1, self.num_heads, self.head_dim)

        h = x @ self.in_proj_weight.t()
        h = h.view(B, T, self.num_heads, self.head_dim * 3)
        w_head_q, w_head_k, w_head_v = h.chunk(3, dim=-1)

        rw_head_q = w_head_q + self.u
        AC = rw_head_q.transpose(1, 2) @ w_head_k.permute(0, 2, 3, 1)

        rr_head_q = w_head_q + self.v
        BD = rr_head_q.transpose(1, 2) @ pos_emb.permute(1, 2, 0)  # [B, H, T, 2T-1]
        BD = F.pad(BD, (1, 1)).view(B, self.num_heads, 2 * T + 1, T)[
            :, :, 1::2, :
        ]  # [B, H, T, T]

        attn_score = (AC + BD) / self.head_dim**0.5

        if mask is not None:
            attn_score = attn_score.masked_fill(mask, -inf)

        with torch.cuda.amp.autocast(enabled=False):
            attn_prob = F.softmax(attn_score.float(), dim=-1)
        attn_prob = F.dropout(attn_prob, self.dropout, self.training)

        attn_vec = attn_prob @ w_head_v.transpose(1, 2)
        return self.out_proj(attn_vec.permute(0, 2, 1, 3).reshape(B, T, -1))


class RelEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, *args, dropout: float = 0.1, **kwargs):
        super().__init__(
            d_model, nhead, *args, batch_first=True, dropout=dropout, **kwargs
        )
        self.self_attn = RelMultiheadAttention(d_model, nhead, dropout=dropout)

    def _sa_block(self, x: Tensor, mask: Tensor = None) -> Tensor:
        return self.dropout1(self.self_attn(x, mask))

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask))
            x = self.norm2(x + self._ff_block(x))
        return x


class RelTDemucs(nn.Module):
    def __init__(
        self,
        in_channels=2,
        num_sources=4,
        channels=64,
        depth=6,
        context_size=None,
        rescale=0.1,
        resample=True,
        kernel_size=8,
        stride=4,
        attention_layers=8,
        **kwargs
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.depth = depth
        self.channels = channels
        self.context_size = context_size

        if resample:
            self.up_sample = Resample(1, 2)
            self.down_sample = Resample(2, 1)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        current_channels = in_channels
        for index in range(depth):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(current_channels, channels, kernel_size, stride),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(channels, channels * 2, 1),
                    nn.GLU(dim=1),
                )
            )

            out_channels = current_channels if index > 0 else num_sources * in_channels

            decode = [
                nn.Conv1d(channels, channels * 2, 3, padding=1, bias=False),
                nn.GLU(dim=1),
                nn.ConvTranspose1d(
                    channels,
                    out_channels,
                    kernel_size,
                    stride,
                ),
            ]
            if index > 0:
                decode.append(nn.ReLU(inplace=True))
            self.decoder.insert(0, nn.Sequential(*decode))
            current_channels = channels
            channels *= 2

        channels = current_channels

        encoder_layer = RelEncoderLayer(
            d_model=channels, dim_feedforward=channels * 4, **kwargs
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, attention_layers)
        self.apply(rescale_conv(reference=rescale))

    def forward(self, x, context_size: int = None):
        batch, ch, _ = x.shape
        mono = x.mean(1, keepdim=True)
        mu = mono.mean(dim=-1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True).add_(1e-5)
        x = standardize(x, mu, std)

        if hasattr(self, "up_sample"):
            x = self.up_sample(x)

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        x = x.transpose(1, 2)
        mask = None
        if context_size is None and self.context_size is not None:
            context_size = self.context_size

        if context_size and context_size < x.size(1):
            mask = x.new_ones(x.size(1), x.size(1), dtype=torch.bool)
            mask = torch.triu(mask, diagonal=context_size)
            mask = mask | mask.T

        x = self.transformer(x, mask).transpose(1, 2)

        for decode in self.decoder:
            skip = saved.pop()
            x = decode(x + skip[..., : x.shape[-1]])

        if hasattr(self, "down_sample"):
            x = self.down_sample(x)

        x = destandardize(x, mu, std)
        x = x.view(batch, -1, ch, x.shape[-1])
        return x
