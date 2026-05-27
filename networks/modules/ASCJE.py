import torch
import torch.nn as nn


class ASCJE(nn.Module):
    def __init__(self, in_ch: int, reduction: int = 16):
        super().__init__()

        hidden = max(8, in_ch // reduction)

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

        self.local = nn.Sequential(
            nn.Conv2d(in_ch, in_ch,kernel_size=3, stride=1, padding=1,groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_ch, kernel_size=1, bias=False),
        )

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)

        k_base = self.proj(x)
        k = k_base + self.local(k_base)

        square = (k - q).pow(2)
        sigma = square.mean(dim=[2, 3], keepdim=True)
        eps = torch.finfo(square.dtype).eps
        var_att = self.sigmoid(square / (2 * sigma + eps) + 0.5)  # (B, C, H, W)

        ch_avg = self.channel_mlp(self.avg_pool(k))
        ch_max = self.channel_mlp(self.max_pool(k))
        ch_att = self.sigmoid(ch_avg + ch_max)  # (B, C, 1, 1)

        sp_avg = torch.mean(k, dim=1, keepdim=True)
        sp_max, _ = torch.max(k, dim=1, keepdim=True)
        sp_att = self.sigmoid(
            self.spatial_conv(torch.cat([sp_avg, sp_max], dim=1))
        )

        att = var_att * ch_att * sp_att
        self.last_att = att.detach()
        out = x * (1.0 + self.gamma * att)
        self.last_out = out.detach()

        return out


