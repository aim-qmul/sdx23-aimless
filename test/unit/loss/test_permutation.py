import pytest
import torch

from loss.permutation import PITLoss


class TestPITLoss:
    @pytest.fixture
    def inner_loss(self):
        return lambda x, y, dims: ((x - y).square().sum(dim=dims).sqrt(), dict())

    @pytest.fixture
    def pit_loss(self, inner_loss):
        return PITLoss(inner_loss)

    def test_correctly_selects_permutation(self, pit_loss):
        x = torch.tensor(
            [
                [
                    [2, 7, 17],
                    [3, 11, 19],
                    [5, 13, 23],
                ]
            ]
        )

        # permute the rows of y
        y = x[:, [2, 0, 1]]

        loss, _ = pit_loss(x, y)
        assert loss == 0.0

    def test_correctly_selects_permutation_across_batches(self, pit_loss):
        batch, channels, bin, step = 11, 5, 7, 13

        x = torch.rand(batch, channels, bin, step)
        y = x[:, torch.randperm(channels)]

        loss, _ = pit_loss(x, y)
        assert loss == 0.0

    def test_adapts_to_differing_numbers_of_dims_before_and_after_channels(
        self, inner_loss
    ):
        n_dims = 6
        shape = torch.randint(3, 8, (n_dims,)).tolist()
        channel_dim = 3

        x = torch.rand(*shape)
        y = x.index_select(channel_dim, torch.randperm(shape[channel_dim]))

        pit_loss = PITLoss(inner_loss, channel_dim=channel_dim)

        loss, _ = pit_loss(x, y)
        assert loss == 0.0
