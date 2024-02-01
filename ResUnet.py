import torch.nn as nn
from torch import cat as cat


# Paper:Road Extraction by Deep Residual U-Net

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt0 = nn.BatchNorm2d(out_ch)
        self.rl0 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.conv_skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        self.bt_skip = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bt0(x0)
        x0 = self.rl0(x0)
        x0 = self.conv1(x0)
        skip = self.conv_skip(x)
        skip = self.bt_skip(skip)
        x1 = x0 + skip
        return x1


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()
        self.bt0 = nn.BatchNorm2d(in_ch)
        self.rl0 = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.bt1 = nn.BatchNorm2d(out_ch)
        self.rl1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.conv_skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, padding=0)
        self.bt_skip = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x0 = self.bt0(x)
        x0 = self.rl0(x0)
        x0 = self.conv0(x0)
        x0 = self.bt1(x0)
        x0 = self.rl1(x0)
        x0 = self.conv1(x0)
        skip = self.conv_skip(x)
        skip = self.bt_skip(skip)
        res = x0 + skip
        return res


class UpB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpB, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.bt0 = nn.BatchNorm2d(out_ch * 2)
        self.rl0 = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, stride=1, padding=1)
        self.bt1 = nn.BatchNorm2d(out_ch)
        self.rl1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.conv_skip = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0)
        self.bt_skip = nn.BatchNorm2d(out_ch)

    def forward(self, x, x_):
        x1 = self.up(x)
        x2 = cat((x1, x_), dim=1)

        x3 = self.bt0(x2)
        x3 = self.rl0(x3)
        x3 = self.conv0(x3)
        x3 = self.bt1(x3)
        x3 = self.rl1(x3)
        x3 = self.conv1(x3)
        skip = self.conv_skip(x2)
        skip = self.bt_skip(skip)
        res = x3 + skip
        return res


class Outcome(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outcome, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.conv(x)
        return x1


class ResUnet(nn.Module):
    def __init__(self):
        super(ResUnet, self).__init__()
        # Encoder
        self.conv0 = ConvBlock(1, 32)
        self.Res1 = ResBlock(32, 64)
        self.Res2 = ResBlock(64, 128)
        # Bridge
        self.Res3 = ResBlock(128, 256)
        # Decoder
        self.up1 = UpB(256, 128)
        self.up2 = UpB(128, 64)
        self.up3 = UpB(64, 32)
        self.outc = Outcome(32, 1)

    def forward(self, x):
        # Encoder
        x1 = self.conv0(x)
        x2 = self.Res1(x1)
        x3 = self.Res2(x2)
        # Bridge
        x4 = self.Res3(x3)
        # Decoder
        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        x7 = self.up3(x6, x1)
        x8 = self.outc(x7)
        return x8
