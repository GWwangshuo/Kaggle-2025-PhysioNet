import auraloss
from torch import nn
from torch.nn import functional as F

def diff(x):
    return x[..., 1:] - x[..., :-1]


class SN_RSTFT_Loss(nn.Module):
    def __init__(
        self,
        snr_weight=1.0,
        stft_weight=1.0,
        diff_weight=1.0,
        stft_kwargs=None,
    ):
        super().__init__()

        self.snr_weight = snr_weight
        self.stft_weight = stft_weight
        self.diff_weight = diff_weight

        self.snr_loss = auraloss.time.SNRLoss()

        if stft_kwargs is None:
            stft_kwargs = {}

        self.stft_loss = auraloss.freq.MultiResolutionSTFTLoss(**stft_kwargs)

    def forward(self, pred, target):
        """
        pred, target: [B, C, T] or [B, T]
        """
        loss_snr = self.snr_loss(pred, target)
        loss_stft = self.stft_loss(pred, target)
        loss_diff = F.l1_loss(diff(pred), diff(target))

        loss = (
            self.snr_weight * loss_snr
            + self.stft_weight * loss_stft
            + self.diff_weight * loss_diff
        )

        return {
            "loss": loss,
            "snr": loss_snr.detach(),
            "mrstft": loss_stft.detach(),
            "diff": loss_diff.detach(),
        }