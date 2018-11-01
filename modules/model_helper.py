import torch
from torch import nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, interpolate=True):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.interpolate = interpolate

    def forward(self, to_up, down):
        if self.interpolate:
            up = F.interpolate(to_up, down.size()[2:], mode="bilinear")
            cropped_down = down
        else:
            up = self.up(to_up)
            cropped_down = self.crop_to_same_size(down, up)
        concatenated_var = torch.cat([cropped_down, up], dim=1)
        return self.double_conv(concatenated_var)

    def crop_to_same_size(self, down, up):
        crop_size = (down.size()[2] - up.size()[2]) // 2
        cropped_down = down[:, :, crop_size:up.size()[2] + crop_size, crop_size:up.size()[2] + crop_size]
        return cropped_down


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch)
        )


    def forward(self, x):
        return self.down(x)