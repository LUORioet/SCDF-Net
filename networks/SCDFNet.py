import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.modules.SCMDConv_ASCJE import SCMDConv_ASCJE
try:
    from thop import profile
    from torchsummary import summary
except Exception:
    profile = None
    summary = None


class First_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(First_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.Conv = nn.Sequential(
            SCMDConv_ASCJE(in_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SCMDConv_ASCJE(out_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.Conv(x)


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3.0) / 6.0


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.hsigmoid = HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.hsigmoid(x)


class DDCoordAtt(nn.Module):
    def __init__(self, inp, oup=None, reduction=16):
        super().__init__()
        if oup is None:
            oup = inp

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HSwish(inplace=True)

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        a_h = a_h.expand(-1, -1, h, w)
        a_w = a_w.expand(-1, -1, h, w)
        return a_w, a_h


class MDEFM(nn.Module):
    def __init__(self, channel_dim, reduction=16):
        super().__init__()
        self.channel_dim = channel_dim

        self.coord_att = DDCoordAtt(channel_dim, channel_dim, reduction=reduction)

        self.connection_branch = nn.Sequential(
            nn.Conv2d(2 * channel_dim, 2 * channel_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * channel_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * channel_dim, channel_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_dim),
            nn.ReLU(inplace=True)
        )

        self.out_norm = nn.Sequential(
            nn.BatchNorm2d(channel_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, f1, f2):
        diff = torch.abs(f1 - f2)
        a_w, a_h = self.coord_att(diff)
        fd = diff * a_w * a_h

        fc = self.connection_branch(torch.cat([f1, f2], dim=1))

        out = self.out_norm(fd + fc)
        return out



class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class GDFDM(nn.Module):
    def __init__(self, high_ch, low_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(high_ch, low_ch, kernel_size=2, stride=2)
        self.gate = ChannelGate(low_ch, reduction=16)
        self.refine = DoubleConv(low_ch, low_ch)

    def forward(self, high, low):
        high = self.up(high)
        if high.shape[2:] != low.shape[2:]:
            high = F.interpolate(high, size=low.shape[2:], mode="bilinear", align_corners=False)

        # gate 越大，越信任低层定位细节；1-gate 越大，越信任高层语义。
        gate = self.gate(low + high)
        fused = low * gate + high * (1.0 - gate)
        return self.refine(fused)


class SCDFNet(nn.Module):
    def __init__(self, in_ch, out_ch, ratio=0.5):
        super(SCDFNet, self).__init__()

        ch1 = int(64 * ratio)
        ch2 = int(128 * ratio)
        ch3 = int(256 * ratio)
        ch4 = int(512 * ratio)
        ch5 = int(1024 * ratio)

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_1 = First_DoubleConv(in_ch, ch1)
        self.Conv1_2 = First_DoubleConv(in_ch, ch1)

        self.Conv2_1 = DoubleConv(ch1, ch2)
        self.Conv2_2 = DoubleConv(ch1, ch2)

        self.Conv3_1 = DoubleConv(ch2, ch3)
        self.Conv3_2 = DoubleConv(ch2, ch3)

        self.Conv4_1 = DoubleConv(ch3, ch4)
        self.Conv4_2 = DoubleConv(ch3, ch4)

        self.Conv5_1 = DoubleConv(ch4, ch5)
        self.Conv5_2 = DoubleConv(ch4, ch5)

        self.mdefm1 = MDEFM(ch1, reduction=16)
        self.mdefm2 = MDEFM(ch2, reduction=16)
        self.mdefm3 = MDEFM(ch3, reduction=16)
        self.mdefm4 = MDEFM(ch4, reduction=16)
        self.mdefm5 = MDEFM(ch5, reduction=16)

        self.UpFuse5 = GDFDM(high_ch=ch5, low_ch=ch4)
        self.UpFuse4 = GDFDM(high_ch=ch4, low_ch=ch3)
        self.UpFuse3 = GDFDM(high_ch=ch3, low_ch=ch2)
        self.UpFuse2 = GDFDM(high_ch=ch2, low_ch=ch1)

        self.Conv_1x1 = nn.Conv2d(ch1, out_ch, kernel_size=1, stride=1, padding=0)
        self.out_act = nn.Sigmoid()

    def forward(self, x1, x2):

        # Encoder
        c1_1 = self.Conv1_1(x1)
        c1_2 = self.Conv1_2(x2)
        d1 = self.mdefm1(c1_1, c1_2)

        c2_1 = self.Conv2_1(self.Maxpool(c1_1))
        c2_2 = self.Conv2_2(self.Maxpool(c1_2))
        d2 = self.mdefm2(c2_1, c2_2)

        c3_1 = self.Conv3_1(self.Maxpool(c2_1))
        c3_2 = self.Conv3_2(self.Maxpool(c2_2))
        d3 = self.mdefm3(c3_1, c3_2)

        c4_1 = self.Conv4_1(self.Maxpool(c3_1))
        c4_2 = self.Conv4_2(self.Maxpool(c3_2))
        d4 = self.mdefm4(c4_1, c4_2)

        c5_1 = self.Conv5_1(self.Maxpool(c4_1))
        c5_2 = self.Conv5_2(self.Maxpool(c4_2))
        d5 = self.mdefm5(c5_1, c5_2)

        # Decoder
        y4 = self.UpFuse5(d5, d4)
        y3 = self.UpFuse4(y4, d3)
        y2 = self.UpFuse3(y3, d2)
        y1 = self.UpFuse2(y2, d1)

        logits = self.Conv_1x1(y1)
        out = self.out_act(logits)
        return out


if __name__ == "__main__":
    AX = torch.randn(1, 3, 256, 256)
    BX = torch.randn(1, 3, 256, 256)
    model = SCDFNet(3, 1, ratio=0.5)
    out_result = model(AX, BX)
    print("Output shape:", out_result.shape)

    if summary is not None:
        summary(model, [(3, 256, 256), (3, 256, 256)])

    if profile is not None:
        flops, params = profile(model, inputs=(AX, BX))
        print("FLOPs:", flops)
        print("Params:", params)
