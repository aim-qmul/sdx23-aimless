import pytest
import torch

from loss.permutation import PITLoss


class TestPITLoss:
    @pytest.fixture
    def inner_loss(self):
        return lambda x, y, dims: (x - y).square().sum(dim=dims).sqrt()

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

        loss = pit_loss(x, y)
        assert loss == 0.0