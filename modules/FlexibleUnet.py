import torch
import torch.nn as nn

from modules.blocks import StackDecoder, StackEncoder, ConvBnRelu2d


class UNet(nn.Module):
    """  More Flexible Unet with different levels """

    def __init__(self, input_channels, level=5, classes=1, is_bn=True):
        super(UNet, self).__init__()
        # save intermediate result
        self.res = []
        self.level = level

        # list for downs and ups
        self.downs = nn.ParameterList()
        self.ups = nn.ParameterList()
        # center operation
        self.center = None

        #  64, 128, 256, 512, 1024
        channels = 64
        self.downs.add_module("down0", StackEncoder(input_channels, channels, kernel_size=3, is_bn=is_bn))
        for i in range(level - 1):
            self.downs.add_module("down" + str(i + 1), StackEncoder(channels, channels * 2, 3, is_bn))
            channels *= 2

        # 1024,
        self.center = ConvBnRelu2d(channels, channels, kernel_size=1, padding=1, stride=1, is_bn=is_bn)

        # 512, 256, 128, 64, 32
        for i in range(level):
            self.ups.add_module("up" + str(i), StackDecoder(channels, channels, channels // 2, kernel_size=3, is_bn=is_bn))
            channels = channels // 2

        # final classifier
        self.classify = nn.Conv2d(channels, classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        out = x
        res_down = []

        # down sample
        for i in range(self.level):
            current_down, out = getattr(self.downs, "down" + str(i))(out)
            res_down.append(current_down)

        # center operation
        out = self.center(out)

        torch.nn.Conv3d
        # up sample
        for i in range(self.level):
            out = getattr(self.ups, "up" + str(i))(res_down[self.level - i - 1], out)

        out = self.classify(out)
        return out
