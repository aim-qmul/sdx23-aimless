import pytest
import torch

from aimless.models import dpt


@pytest.fixture(params=[10])
def embedding_size(request):
    return request.param


@pytest.fixture
def dpt_layer(embedding_size):
    return dpt.DPTLayer(embedding_size, 2, 20, 0.1, "ReLU")


def test_dpt_layer_forward(dpt_layer, embedding_size):
    batch_size = 10
    seq_len = 20
    inputs = torch.testing.make_tensor(
        batch_size, seq_len, embedding_size, dtype=torch.float32, device="cpu"
    )

    outputs = dpt_layer(inputs)

    assert outputs.shape == (batch_size, seq_len, embedding_size)


def test_dpt_layer_transforms_input(dpt_layer, embedding_size):
    batch_size = 11
    seq_len = 21
    inputs = torch.testing.make_tensor(
        batch_size, seq_len, embedding_size, dtype=torch.float32, device="cpu"
    )

    outputs = dpt_layer(inputs)

    assert not torch.allclose(inputs, outputs)


@pytest.mark.parametrize("activation", ["ReLU", "GELU", "ELU", "LeakyReLU", "Tanh"])
def test_dpt_layer_uses_selected_activation(mocker, activation):
    spy = mocker.spy(dpt.torch.nn, activation)
    dpt.DPTLayer(10, 2, 20, 0.1, activation)
    spy.assert_called_once()


def test_dpt_filterbank_forward():
    in_channels = 1
    num_filters = 2
    kernel_size = 3

    model = dpt.DPTFilterbank(in_channels, num_filters, kernel_size)

    batch_size = 10
    seq_len = 20
    inputs = torch.testing.make_tensor(
        batch_size, in_channels, seq_len, dtype=torch.float32, device="cpu"
    )

    outputs = model(inputs)

    assert outputs.shape == (batch_size, num_filters, seq_len - 2 * (kernel_size // 2))


def test_dpt_filterbank_forward_transposed():
    in_channels = 1
    num_filters = 2
    kernel_size = 3

    model = dpt.DPTFilterbank(num_filters, in_channels, kernel_size, transpose=True)

    batch_size = 10
    seq_len = 20
    inputs = torch.testing.make_tensor(
        batch_size, num_filters, seq_len, dtype=torch.float32, device="cpu"
    )

    outputs = model(inputs)

    assert outputs.shape == (batch_size, in_channels, seq_len + 2 * (kernel_size // 2))


def test_dpt_forward():
    batch_size = 3
    input_channels = 1
    seq_len = 32

    num_sources = 3
    num_filters = 4
    filter_size = 3

    segment_size = 4
    segment_stride = 2

    num_dual_path_layers = 2
    num_attention_heads = 2

    lstm_hidden_size = 16

    model = dpt.DPT(
        channels=input_channels,
        num_sources=num_sources,
        num_filters=num_filters,
        filter_size=filter_size,
        segment_size=segment_size,
        segment_stride=segment_stride,
        num_dual_path_layers=num_dual_path_layers,
        num_attention_heads=num_attention_heads,
        lstm_hidden_size=lstm_hidden_size,
    )

    x = torch.testing.make_tensor(
        batch_size, input_channels, seq_len, dtype=torch.float32, device="cpu"
    )

    y = model(x)
