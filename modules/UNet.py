import torch
import torch.nn as nn
from modules.blocks import StackDecoder, StackEncoder, ConvBnRelu2d


class UNet(nn.Module):
    def __init__(self, in_shape, is_bn=False):
        super(UNet, self).__init__()
        C, H, W = in_shape

        self.down1 = StackEncoder(C, 64, kernel_size=3, is_bn=is_bn)  # 128
        self.down2 = StackEncoder(64, 128, 3, is_bn)    # 64
        self.down3 = StackEncoder(128, 256, 3, is_bn)   # 32
        self.down4 = StackEncoder(256, 512, 3, is_bn)   # 16
        self.down5 = StackEncoder(512, 1024, 3, is_bn)  # 8

        self.center = ConvBnRelu2d(1024, 1024, kernel_size=1, padding=1, stride=1, is_bn=is_bn)

        # 8
        # x_big_channels, x_channels, y_channels
        self.up5 = StackDecoder(1024, 1024, 512, kernel_size=3, is_bn=is_bn)    # 16
        self.up4 = StackDecoder(512, 512, 256, kernel_size=3, is_bn=is_bn)   # 32
        self.up3 = StackDecoder(256, 256, 128, kernel_size=3, is_bn=is_bn)  # 64
        self.up2 = StackDecoder(128, 128, 64, kernel_size=3, is_bn=is_bn)   # 128
        self.up1 = StackDecoder(64, 64, 32, kernel_size=3, is_bn=is_bn) # 256

        self.classify = nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        out = x
        down1, out = self.down1(out)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)

        out = self.center(out)
        # there should merge out with down4 not down5
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)

        out = self.classify(out)
        return out
