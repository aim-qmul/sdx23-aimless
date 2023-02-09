import torch
from torch import nn

__all__ = ["DPTLayer"]


def _get_activation(activation: str):
    try:
        return getattr(torch.nn, activation)()
    except AttributeError:
        raise ValueError(f"Activation {activation} not supported.")


class DPTLayer(nn.Module):
    """Implements a layer of the Dual-Path Transformer, as described in [1]

    Args:
        embedding_size (int): The size of the input embedding.
        num_heads (int): The number of attention heads.
        hidden_size (int): The size of the hidden layer in the LSTM.
        dropout (float): The dropout rate.
        activation (str): The activation function to use. One of ["relu", "gelu"].
        **lstm_kwargs: Additional keyword arguments to pass to the LSTM.

    References:
        [1] https://arxiv.org/abs/2007.13975
    """

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        hidden_size: int,
        dropout: float,
        activation: str,
        **lstm_kwargs,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embedding_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_size)

        self.lstm = nn.LSTM(
            embedding_size, hidden_size, batch_first=True, **lstm_kwargs
        )
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, embedding_size), _get_activation(activation)
        )
        self.norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(x, x, x)[0]
        x = self.norm1(x)

        h, _ = self.lstm(x)
        x = x + self.dense(h)
        x = self.norm2(x)

        return x
