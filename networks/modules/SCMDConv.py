import math
import torch
from torch import nn
from networks.modules.RMDConv import RMDConv


class SCMDConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, ratio=2, aux_k=3, dilation=3):
        super(SCMDConv, self).__init__()
        self.out_ch = out_ch
        native_ch = math.ceil(out_ch / ratio)
        aux_ch = native_ch * (ratio - 1)

        self.native = nn.Sequential(
            nn.Conv2d(in_ch, native_ch, kernel_size, stride, padding=padding, dilation=1, bias=False),
            nn.BatchNorm2d(native_ch),
            nn.ReLU(inplace=True),
        )

        self.aux = nn.Sequential(
            RMDConv(native_ch, aux_ch, aux_k, 1, padding=1, groups=int(native_ch / 4), dilation=dilation,
                   bias=False),
            nn.BatchNorm2d(aux_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.native(x)
        x2 = self.aux(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_ch, :, :]
