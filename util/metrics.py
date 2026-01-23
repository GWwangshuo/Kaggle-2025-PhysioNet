import torch

@torch.no_grad()
def snr_metric(y_pred, y_gt, eps=1e-7):
    """
    y_pred, y_gt: [B, C, T] or [B, T]
    return: float (mean SNR in dB)
    """
    signal = torch.mean(y_gt ** 2, dim=-1)
    noise  = torch.mean((y_pred - y_gt) ** 2, dim=-1)

    snr_db = 10 * torch.log10((signal + eps) / (noise + eps))

    return snr_db.mean()