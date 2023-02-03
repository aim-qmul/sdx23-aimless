import math
from typing import Callable

from lapsolver import solve_dense
import torch


class PITLoss(torch.nn.Module):
    def __init__(self, loss: Callable, channel_dim: int = 1) -> None:
        super().__init__()

        self.loss = loss
        self.channel_dim = channel_dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_dims = x.shape[: self.channel_dim]
        batch = math.prod(batch_dims)

        x_ = x.unsqueeze(self.channel_dim).reshape(
            batch, 1, *x.shape[self.channel_dim :]
        )
        y_ = y.unsqueeze(self.channel_dim + 1).reshape(
            batch, -1, 1, *y.shape[self.channel_dim + 1 :]
        )
        dims = list(range(3, x_.ndim))

        # compute the distance matrix
        dists, extra = self.loss(x_, y_, dims)

        # solve the linear assignment problem
        assignments = [solve_dense(dists[b]) for b in range(batch)]
        losses = []
        for b in range(batch):
            rows, cols = assignments[b]
            losses.append(dists[b, rows, cols].sum())

        return sum(losses), extra | dict(assignments=assignments)
