import torch
import torch.nn as nn
import numpy as np

from modules.blocks import StackDecoder, StackEncoder, ConvBnRelu


class UNet(nn.Module):
    """  More Flexible Unet with different levels """

    def __init__(self, input_channels, level=5, classes=1, is_bn=True, d3=False):
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
        self.downs.add_module("down0", StackEncoder(input_channels, channels, kernel_size=3, is_bn=is_bn, d3=d3))
        for i in range(level - 1):
            self.downs.add_module("down" + str(i + 1), StackEncoder(channels, channels * 2, 3, is_bn, d3=d3))
            channels *= 2

        # 1024,
        self.center = ConvBnRelu(channels, channels, kernel_size=1, padding=1, stride=1, is_bn=is_bn, d3=d3)

        # 512, 256, 128, 64, 32
        for i in range(level):
            self.ups.add_module("up" + str(i), StackDecoder(channels, channels, channels // 2, kernel_size=3, is_bn=is_bn, d3=d3))
            channels = channels // 2

        # final classifier
        if d3:
            self.classify = nn.Conv3d(channels, classes, kernel_size=1, padding=0, stride=1, bias=True)
        else:
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

def test_2d_flexible_unet():
    net = UNet(1, level=3)
    a = torch.Tensor(np.random.randn(1, 1, 128, 128))
    res = net(a)
    print(res.shape)


def test_3d_unet():
    net = UNet(1, level=2, d3=True)
    a = torch.Tensor(np.random.randn(1, 1, 10, 128, 128))
    res = net(a)
    print(res.shape)

if __name__ == '__main__':
    test_2d_flexible_unet()
    test_3d_unet()