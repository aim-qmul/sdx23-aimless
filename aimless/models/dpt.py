from typing import Callable

from einops import rearrange
import torch
from torch import nn


def _get_activation(activation: str):
    try:
        return getattr(torch.nn, activation)()
    except AttributeError:
        raise ValueError(f"Activation {activation} not supported.")


class DPTFilterbank(nn.Module):
    """A Conv-TasNet style filterbank for DPT.

    Args:
        input_channels (int): The number of input channels.
        num_filters (int): The number of filters to learn.
        kernel_size (int): The size of the filters.
        nonlinearity (str): The nonlinearity to use. One of ["ReLU", "GELU", "ELU",
            "LeakyReLU", "Tanh"].
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_filters: int = 64,
        kernel_size: int = 16,
        nonlinearity: str = "ReLU",
        transpose: bool = False,
    ):
        super().__init__()

        _conv_module = nn.ConvTranspose1d if transpose else nn.Conv1d

        self.filterbank = _conv_module(
            input_channels,
            num_filters,
            kernel_size,
            stride=kernel_size // 2,
            padding=0,
        )
        self.nonlinearity = (
            _get_activation(nonlinearity) if nonlinearity is not None else lambda x: x
        )

    def forward(self, x: torch.Tensor):
        return self.nonlinearity(self.filterbank(x))


class DPTLayer(nn.Module):
    """One layer of the Dual-Path Transformer, as described in [1]

    Args:
        embedding_size (int): The size of the input embedding.
        num_heads (int): The number of attention heads.
        hidden_size (int): The size of the hidden layer in the LSTM.
        dropout (float): The dropout rate.
        activation (str): The activation function to use in `nn.modules.activations`.
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
        bidirectional: bool = True,
        **lstm_kwargs,
    ):
        super().__init__()

        if embedding_size % num_heads != 0:
            raise ValueError(
                f"Embedding size {embedding_size} must be divisible by number of "
                f"heads {num_heads}."
            )

        self.attention = nn.MultiheadAttention(
            embedding_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_size)

        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
            **lstm_kwargs,
        )
        self.dense = nn.Sequential(
            nn.Linear(
                hidden_size * 2 if bidirectional else hidden_size, embedding_size
            ),
            _get_activation(activation),
        )
        self.norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(x, x, x)[0]
        x = self.norm1(x)

        h, _ = self.lstm(x)
        x = x + self.dense(h)
        x = self.norm2(x)

        return x


class DualPathLayer(nn.Module):
    """Apply intra- and inter-chunk transformer layers.

    Args:
        embedding_size (int): The size of the input embedding.
        num_heads (int): The number of attention heads.
        hidden_size (int): The size of the hidden layer in the LSTM.
        dropout (float): The dropout rate.
        activation (str): The activation function to use in `nn.modules.activations`.
        **lstm_kwargs: Additional keyword arguments to pass to the LSTM.
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
        self.intra_chunk = DPTLayer(
            embedding_size, num_heads, hidden_size, dropout, activation, **lstm_kwargs
        )
        self.inter_chunk = DPTLayer(
            embedding_size, num_heads, hidden_size, dropout, activation, **lstm_kwargs
        )

    def _apply_intra_chunk(self, x: torch.Tensor, fn: Callable):
        batch_size, *_ = x.shape
        x = rearrange(x, "b c m n -> (b n) m c")
        x = fn(x)
        x = rearrange(x, "(b n) m c -> b c m n", b=batch_size)
        return x

    def _apply_inter_chunk(self, x: torch.Tensor, fn: Callable):
        batch_size, *_ = x.shape
        x = rearrange(x, "b c m n -> (b m) n c")
        x = fn(x)
        x = rearrange(x, "(b m) n c -> b c m n", b=batch_size)
        return x

    def forward(self, x: torch.Tensor):
        x = self._apply_intra_chunk(x, self.intra_chunk)
        x = self._apply_inter_chunk(x, self.inter_chunk)
        return x


class DPT(nn.Module):
    """The Dual-Path Transformer, as described in [1].

    Args:
        nn (_type_): _description_

    References:
        [1] https://arxiv.org/abs/2007.13975
    """

    def __init__(
        self,
        channels: int = 2,
        num_sources: int = 4,
        num_filters: int = 64,
        filter_size: int = 16,
        filterbank_nonlinearity: str = "ReLU",
        segment_size: int = 100,
        segment_stride: int = 50,
        num_dual_path_layers: int = 6,
        num_attention_heads: int = 4,
        lstm_hidden_size: int = 256,
        transformer_dropout: float = 0.1,
        transformer_nonlinearity: str = "GELU",
        post_transformer_prelu: bool = True,
        mask_nonlinearity: str = "ReLU",
    ):
        super().__init__()
        self.num_sources = num_sources

        self.encoder = DPTFilterbank(
            input_channels=channels,
            num_filters=num_filters,
            kernel_size=filter_size,
            nonlinearity=filterbank_nonlinearity,
        )
        self.decoder = DPTFilterbank(
            input_channels=num_filters,
            num_filters=channels,
            kernel_size=filter_size,
            nonlinearity=None,
            transpose=True,
        )
        self.pre_norm = nn.LayerNorm(num_filters)
        self.segment_size = segment_size
        self.segment_stride = segment_stride

        transformer_net = []
        for _ in range(num_dual_path_layers):
            transformer_net.append(
                DualPathLayer(
                    num_filters,
                    num_attention_heads,
                    lstm_hidden_size,
                    dropout=transformer_dropout,
                    activation=transformer_nonlinearity,
                    bidirectional=True,
                )
            )
        self.transformer_net = nn.Sequential(*transformer_net)

        post_transformer = []
        if post_transformer_prelu:
            post_transformer.append(nn.PReLU())
        post_transformer.append(nn.Conv2d(num_filters, num_sources * num_filters, 1))
        self.post_transformer = nn.Sequential(*post_transformer)

        self.gate_paths = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(num_filters, num_filters, 1), nn.Tanh()),
                nn.Sequential(nn.Conv1d(num_filters, num_filters, 1), nn.Sigmoid()),
            ]
        )

        self.mask_activation = _get_activation(mask_nonlinearity)

    def _global_norm(self, x: torch.Tensor):
        return x / x.norm(dim=1, keepdim=True)

    def _segment(self, x: torch.Tensor):
        x = rearrange(x, "b c t -> b c t ()")
        x_segmented = nn.functional.unfold(
            x,
            kernel_size=(self.segment_size, 1),
            stride=(self.segment_stride, 1),
            padding=(self.segment_size, 0),
        )
        x = rearrange(x_segmented, "b (c m) n -> b c m n", m=self.segment_size)
        return x

    def _unsegment(self, x: torch.Tensor, original_len: int):
        x = rearrange(x, "b c m n -> b (c m) n")
        x = nn.functional.fold(
            x,
            output_size=(original_len, 1),
            kernel_size=(self.segment_size, 1),
            stride=(self.segment_stride, 1),
            padding=(self.segment_size, 0),
        )
        x = rearrange(x, "b c t () -> b c t")
        return x

    def forward(self, x: torch.Tensor):
        # apply input filterbank
        x = self.encoder(x)

        # preserve shap for unsegmenting later
        *_, original_len = x.shape

        # pre-normalisation
        m = self.pre_norm(x.transpose(1, 2)).transpose(1, 2)

        # perform segmentation
        m = self._segment(x)

        # apply transformer
        m = self.transformer_net(m)

        # project to high dimension
        m = self.post_transformer(m)
        m = rearrange(m, "b (s c) m n -> (b s) c m n", s=self.num_sources)

        # unsegment
        m = self._unsegment(m, original_len)

        # apply gating
        m = [g(m) for g in self.gate_paths]
        m = torch.mul(*m)

        # reshape to recover masks
        m = rearrange(m, "(b s) c t -> b s c t", s=self.num_sources)
        m = self.mask_activation(m)

        # apply masks
        y_ = m * rearrange(x, "b c t -> b () c t")

        # move sources to batch dimension and apply transposed filterbank
        y_ = rearrange(y_, "b s c t -> (b s) c t")
        y_ = self.decoder(y_)
        y_ = rearrange(y_, "(b s) c t -> b s c t", s=self.num_sources)

        return y_
