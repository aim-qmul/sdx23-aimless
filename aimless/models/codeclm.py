import torch
from typing import List


class CodecLM(torch.nn.Module):
    def __init__(self, sources: List[str], n_q: int, card: int, dim: int = 128) -> None:
        super().__init__()
        self.sources = sources
        self.n_src = len(sources)

        self.emb = torch.nn.ModuleList(
            [torch.nn.Embedding(card + 1, dim) for _ in range(n_q)]
        )

        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                batch_first=True,
            ),
            num_layers=6,
        )

        self.outputs = torch.nn.ModuleList()
        for n in range(self.n_src):
            self.outputs.append(
                torch.nn.ModuleList(
                    [torch.nn.Linear(dim, card + 1) for _ in range(n_q)]
                )
            )

        self.logsoftmax = torch.nn.LogSoftmax(dim=1)  # dim over classes

    def forward(self, x: torch.Tensor):
        B, K, T = x.size()
        input_ = sum([self.emb[k](x[:, k]) for k in range(K)])
        print(input_.shape)

        out = self.transformer(input_)
        print(out.shape)

        outputs = []
        for n in range(self.n_src):
            outputs.append(
                torch.stack([self.outputs[n][k](out) for k in range(K)], dim=1)
            )

        outputs = torch.stack(outputs, dim=1)
        print(outputs.shape)
        outputs = outputs.permute(0, 4, 1, 2, 3)  # put classes on dim=1
        # shape: (bs, codes, src, codebooks, frames)
        print(outputs.shape)
        outputs = self.logsoftmax(outputs)
        print(outputs.shape)
        return outputs
