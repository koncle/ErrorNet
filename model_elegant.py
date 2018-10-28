import torch
import torch.nn as nn

from model_helper import DoubleConv, Up, Down

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        self.DEBUG = False
        self.down0 = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.dropout3 = nn.Dropout2d(0.5)
        self.down4 = Down(512, 1024)
        self.dropout4 = nn.Dropout2d(0.5)

        self.up3 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up1 = Up(256, 128)
        self.up0 = Up(128, 64)

        self.conv_out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        down0 = self.down0(X)
        down1 = self.down1(down0)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(self.dropout3(down3))
        up3 = self.up3(self.dropout4(down4), down3)
        up2 = self.up2(up3, down2)
        up1 = self.up1(up2, down1)
        up0 = self.up0(up1, down0)
        out_value = self.conv_out(up0)

        if DEBUG:
            print(down0.size())
            print(down1.size())
            print(down2.size())
            print(down3.size())
            print(down4.size())
            print(up3.size())
            print(up2.size())
            print(up1.size())
            print(up0.size())
            print(out_value.size())
        return self.softmax(out_value)

if __name__ == '__main__':
    unet = UNet(1, 2).cuda()
    a = torch.randn((1, 1, 572, 572)).cuda()
    print(unet(a).size())





