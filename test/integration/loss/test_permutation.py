import torch

from loss.permutation import PITLoss

def test_works_with_freq_loss(self):
    from loss.freq import CLoss

    pit_loss = PITLoss(CLoss())
    x = torch.rand(3, 5, 7, 11)
    y = x[:, torch.randperm(5)]

    loss, _ = pit_loss(x, y)
    assert loss == 0.0