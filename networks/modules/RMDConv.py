import torch
import torch.nn as nn


class RMDConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=3, groups=1, dilation_set=4,
                 bias=False):
        super().__init__()

        self.prim = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=dilation, dilation=dilation,
                              groups=groups * dilation_set, bias=bias)

        self.prim_shift = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=2 * dilation, dilation=2 * dilation,
                                    groups=groups * dilation_set, bias=bias)

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias)

        def backward_hook(grad):
            out = grad.clone()
            mask = self.mask.to(out.device)
            out = out.masked_fill(mask, 0)
            return out

        self.mask = torch.zeros_like(self.conv.weight, dtype=torch.bool, device=self.conv.weight.device)

        _in_channels = in_ch // (groups * dilation_set)
        _out_channels = out_ch // (groups * dilation_set)

        for i in range(dilation_set):
            for j in range(groups):
                self.mask[(i + j * groups) * _out_channels: (i + j * groups + 1) * _out_channels,
                i * _in_channels: (i + 1) * _in_channels, :, :] = True
                self.mask[((i + dilation_set // 2) % dilation_set + j * groups) *
                          _out_channels: ((i + dilation_set // 2) % dilation_set + j * groups + 1) * _out_channels,
                i * _in_channels: (i + 1) * _in_channels, :, :] = True

        self.conv.weight.data.masked_fill_(self.mask, 0)

        self.conv.weight.register_hook(backward_hook)
        self.groups = groups

    def forward(self, x):
        x_split = (z.chunk(2, dim=1) for z in x.chunk(self.groups, dim=1))
        x_merge = torch.cat(tuple(torch.cat((x2, x1), dim=1) for (x1, x2) in x_split), dim=1)
        x_shift = self.prim_shift(x_merge)
        out = self.prim(x) + self.conv(x) + x_shift
        return out
