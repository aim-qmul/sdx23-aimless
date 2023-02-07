import torch
import torch.nn.functional as F
from itertools import combinations, chain


class TLoss(torch.nn.Module):
    r"""Base class for time domain loss modules.
    You can't use this module directly.
    Your loss should also subclass this class.
    Args:
        pred (Tensor): predict time signal tensor with size (batch, *, channels, samples), `*` is an optional multi targets dimension.
        gt (Tensor): target time signal tensor with size (batch, *, channels, samples), `*` is an optional multi targets dimension.
        mix (Tensor): mixture time signal tensor with size (batch, channels, samples)
    Returns:
        tuple: a length-2 tuple with the first element is the final loss tensor,
            and the second is a dict containing any intermediate loss value you want to monitor
    """

    def forward(self, *args, **kwargs):
        return self._core_loss(*args, **kwargs)

    def _core_loss(self, pred, gt, mix):
        raise NotImplementedError


class SDR(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.expr = "bi,bi->b"

    def _batch_dot(self, x, y):
        return torch.einsum(self.expr, x, y)

    def forward(self, estimates, references):
        if estimates.dtype != references.dtype:
            estimates = estimates.to(references.dtype)
        length = min(references.shape[-1], estimates.shape[-1])
        references = references[..., :length].reshape(references.shape[0], -1)
        estimates = estimates[..., :length].reshape(estimates.shape[0], -1)

        delta = 1e-7  # avoid numerical errors
        num = self._batch_dot(references, references)
        den = (
            num
            + self._batch_dot(estimates, estimates)
            - 2 * self._batch_dot(estimates, references)
        )
        den = den.relu().add_(delta).log10()
        num = num.add_(delta).log10()
        return 10 * (num - den)


class CL1Loss(TLoss):
    def _core_loss(self, pred, gt, mix):
        gt = gt[..., : pred.shape[-1]]
        loss = []
        for c in chain(
            combinations(range(4), 1),
            combinations(range(4), 2),
            combinations(range(4), 3),
        ):
            x = sum([pred[:, i] for i in c])
            y = sum([gt[:, i] for i in c])
            loss.append(F.l1_loss(x, y))

        # All 14 Combination Losses (4C1 + 4C2 + 4C3)
        loss_l1 = sum(loss) / len(loss)
        return loss_l1, {}


class L1Loss(TLoss):
    def _core_loss(self, pred, gt, mix):
        return F.l1_loss(pred, gt[..., : pred.shape[-1]]), {}
