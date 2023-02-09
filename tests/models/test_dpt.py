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
