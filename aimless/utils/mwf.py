import torch
import norbert


class MWF(torch.nn.Module):
    def __init__(
        self, residual_model=False, softmask=False, alpha=1.0, n_iter=1
    ) -> None:
        super().__init__()
        self.residual_model = residual_model
        self.n_iter = n_iter
        self.softmask = softmask
        self.alpha = alpha

    def forward(self, msk_hat, mix_spec):
        assert msk_hat.ndim > mix_spec.ndim
        if not self.softmask and not self.n_iter and not self.residual_model:
            return msk_hat * mix_spec.unsqueeze(1)

        V = msk_hat * mix_spec.abs().unsqueeze(1)
        if self.softmask and self.alpha != 1:
            V = V.pow(self.alpha)

        X = mix_spec.transpose(1, 3).contiguous()
        V = V.permute(0, 4, 3, 2, 1).contiguous()

        if self.residual_model or V.shape[4] == 1:
            V = norbert.residual_model(V, X, self.alpha if self.softmask else 1)

        Y = norbert.wiener(
            V, X.to(torch.complex128), self.n_iter, use_softmask=self.softmask
        ).to(X.dtype)

        Y = Y.permute(0, 4, 3, 2, 1).contiguous()
        return Y
