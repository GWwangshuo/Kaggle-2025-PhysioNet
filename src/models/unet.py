import torch
from torch import nn
from typing import List
from src.utils.losses import SN_RSTFT_Loss
from torch.utils.checkpoint import checkpoint

def snr_loss(y_pred, y_gt, eps=1e-7):
    """
    y_pred, y_gt: [B, C, T] or [B, T]
    return: scalar tensor
    """
    signal = torch.mean(y_gt ** 2, dim=-1)
    noise  = torch.mean((y_pred - y_gt) ** 2, dim=-1)

    snr_db = 10 * torch.log10((signal + eps) / (noise + eps))
    return snr_db.mean()

    
class MaskEmbeddingToLeadSignalSoftArgmax(nn.Module):
    def __init__(self, n_leads=4, n_sample=4352, embedding_dim=32, temperature=0.5):
        super().__init__()
        self.n_leads = n_leads
        self.temperature = temperature

        self.lead_y_logits = nn.Conv2d(embedding_dim, n_leads, kernel_size=1)

    def forward(self, masked_feat, mask=None):
        B, C, H, W = masked_feat.shape
        device = masked_feat.device
        dtype = masked_feat.dtype

        y_logits = self.lead_y_logits(masked_feat)
        prob = torch.softmax(y_logits / self.temperature, dim=2)

        y_coord = torch.arange(H, device=device, dtype=dtype).view(1,1,H,1)
        y_pixel = (prob * y_coord).sum(dim=2)  # [B, L, W]

        return y_pixel, prob
            
    def resample_torch(self, x, num, dim=-1):
        """基于 FFT 的 PyTorch 重采样"""
        dim = (x.dim() + dim) if dim < 0 else dim
        X = torch.fft.fft(x, dim=dim)
        Nx = X.shape[dim]

        sl = [slice(None)] * X.ndim
        newshape = list(X.shape)
        newshape[dim] = num
        Y = torch.zeros(newshape, dtype=X.dtype, device=X.device)

        N = min(num, Nx)
        sl[dim] = slice(0, (N + 1) // 2)
        Y[sl] = X[sl]
        sl[dim] = slice(-(N - 1) // 2, None)
        Y[sl] = X[sl]

        if N % 2 == 0:
            if N < Nx:
                sl[dim] = slice(N//2, N//2+1)
                Y[sl] += X[sl]
            elif N < num:
                sl[dim] = slice(num-N//2, num-N//2+1)
                Y[sl] /= 2
                temp = Y[sl]
                sl[dim] = slice(N//2, N//2+1)
                Y[sl] = temp

        y = torch.fft.ifft(Y, dim=dim).real * (float(num) / float(Nx))
        return y
    
    
class UNet(nn.Module):
    def __init__(
        self,
        num_in_channels: int,
        num_out_channels: int,
        depth: int,
        dims: List[int],
        use_checkpoint=True
    ):
        super(UNet, self).__init__()
        
        self.use_checkpoint = use_checkpoint

        self.depth = depth
        self.dims = dims

        # Encoder blocks
        self.encoders = nn.ModuleList(
            [
                self._make_encoder_block(num_in_channels if i == 0 else dims[i - 1], dims[i], depth)
                for i in range(len(dims))
            ]
        )
        self.encoder_skips = nn.ModuleList(
            [self._make_skip_connection(num_in_channels if i == 0 else dims[i - 1], dims[i]) for i in range(len(dims))]
        )
        self.encoder_downscaling = nn.ModuleList(
            [
                nn.Conv2d(
                    dims[i],
                    dims[i],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                )
                for i in range(len(dims))
            ]
        )

        # Decoder blocks
        self.decoders = nn.ModuleList(
            [self._make_decoder_block(dims[i] + dims[i - 1], dims[i - 1]) for i in range(len(dims) - 1, 0, -1)]
        )
        
        self.signal_head = MaskEmbeddingToLeadSignalSoftArgmax()
   
        self.signal_loss = SN_RSTFT_Loss()
        
        # -------------------------------
        # constants (pixel space)
        # -------------------------------
        self.register_buffer(
            'zero_mv',
            torch.tensor([703.5, 987.5, 1271.5, 1531.5]).view(1, 4, 1)
        )  # [1, 4, 1]

        self.register_buffer(
            'mv_to_pixel',
            torch.tensor(79.0)
        )  # scalar tensor
        

    def _enc_forward(self, encoder, skip, x):
        return encoder(x) + skip(x)
    
    def _dec_forward(self, decoder, x):
        return decoder(x)

    def forward(
        self, 
        x: torch.Tensor, 
        ecg_mv: dict[str, torch.Tensor] | None = None,
        target_len: int = None
    ):
        skips = []

        if target_len is not None:
            x = self.signal_head.resample_torch(x, target_len, dim=-1)

        # ---------- Encoder ----------
        for encoder, skip, down in zip(
            self.encoders, self.encoder_skips, self.encoder_downscaling
        ):
            if self.use_checkpoint and self.training:
                x = checkpoint(self._enc_forward, encoder, skip, x, use_reentrant=False)
            else:
                x = encoder(x) + skip(x)

            skips.append(x)
            x = down(x)

        skips = skips[::-1]

        # ---------- Decoder ----------
        for i, decoder in enumerate(self.decoders):
            x = self._upsample(x, skips[i + 1])
            x = torch.cat([x, skips[i + 1]], dim=1)

            if self.use_checkpoint and self.training:
                x = checkpoint(self._dec_forward, decoder, x, use_reentrant=False)
            else:
                x = decoder(x)

        # ---------- Head ----------
        y_pixel, prob = self.signal_head(x)
        pred_mv = (self.zero_mv - y_pixel) / self.mv_to_pixel
        
        _, _, H, L = prob.shape
        
        if ecg_mv is not None:
            gt_mv = ecg_mv[f'{L}']
            
            out = self.signal_loss(pred_mv, gt_mv)
            
            snr = snr_loss(pred_mv, gt_mv)
            out['metric_snr'] = snr
            return out

        return {
            'prob': prob,
            f'y_mv_{L}': pred_mv
        }

    def _make_encoder_block(self, in_channels: int, num_out_channels: int, num_layers: int) -> nn.Sequential:
        layers = [self._conv_norm_act(in_channels, num_out_channels)]
        for _ in range(num_layers - 1):
            layers.append(self._conv_norm_act(num_out_channels, num_out_channels))
        return nn.Sequential(*layers)

    def _make_decoder_block(self, in_channels: int, num_out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            self._conv_norm_act(in_channels, num_out_channels),
            self._conv_norm_act(num_out_channels, num_out_channels),
        )

    def _make_skip_connection(self, in_channels: int, num_out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                num_out_channels,
                kernel_size=1,
                bias=False,
                padding_mode="replicate",
            ),
        )

    def _conv_norm_act(self, in_channels: int, num_out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                num_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode="replicate",
            ),
            nn.InstanceNorm2d(num_out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def _upsample(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return x