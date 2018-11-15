import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPS = 1e-4


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True,
                 is_relu=True, d3=False):
        """
        Convolution + Batch Norm + Relu for 2D feature maps
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param padding:
        :param dilation:
        :param stride:
        :param groups:
        :param is_bn:
        :param is_relu:
        """
        super(ConvBnRelu, self).__init__()
        if d3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=True)
            self.bn = nn.BatchNorm3d(out_channels, eps=BN_EPS)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)

        self.relu = nn.ReLU(inplace=True)
        if is_bn is False:
            self.bn = None

        if is_relu is False:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)
        return x

    def merge_bn(self):
        if self.bn is None:
            return

        assert (self.conv.bias is None)

        conv_weight = self.conv.weight.data
        bn_weight = self.bn.weight.data
        bn_bias = self.bn.bias.data
        bn_running_mean = self.bn.running_mean
        bn_running_var = self.bn.running_var
        bn_eps = self.bn.eps

        # https://github.com/sanghoon/pva-faster-rcnn/issues/5
        # https://github.com/sanghoon/pva-faster-rcnn/commit/39570aab8c6513f0e76e5ab5dba8dfbf63e9c68c

        N, C, KH, KW = conv_weight.size()
        std = 1 / (torch.sqrt(bn_running_var + bn_eps))
        std_bn_weight = (std * bn_weight).repeat(C * KH * KW, 1).t().contiguous().view(N, C, KH, KW)
        conv_weight_hat = std_bn_weight * conv_weight
        conv_bias_hat = (bn_bias - bn_weight * std * bn_running_mean)

        self.bn = None
        self.conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                              kernel_size=self.conv.kernel_size,
                              padding=self.conv.padding, stride=self.conv.stride, dilation=self.conv.dilation,
                              groups=self.conv.groups,
                              bias=True)
        self.conv.weight.data = conv_weight_hat  # fill in
        self.conv.bias.data = conv_bias_hat


class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size, is_bn=True, d3=False):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                       groups=1, is_bn=is_bn, d3=d3),
            ConvBnRelu(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                       groups=1, is_bn=is_bn, d3=d3)
        )
        if d3:
            self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        else:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.encode(x)
        x_small = self.max_pool(x) #F.max_pool2d(x, kernel_size=2, stride=2)
        return x, x_small


class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3, is_bn=True, d3=False):
        super(StackDecoder, self).__init__()
        self.d3 = d3
        padding = (kernel_size - 1) // 2

        self.decode = nn.Sequential(
            ConvBnRelu(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1,
                       stride=1, groups=1, is_bn=is_bn, d3=d3),
            ConvBnRelu(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                       groups=1, is_bn=is_bn, d3=d3)
        )

    def forward(self, x_big, x):
        if self.d3:
            N, C, D, H, W = x_big.size()
            y = F.interpolate(x, size=(D, H, W), align_corners=False, mode='trilinear')
        else:
            N, C, H, W = x_big.size()
            y = F.interpolate(x, size=(H, W), align_corners=False, mode='bilinear')
        y = torch.cat((y, x_big), 1)
        y = self.decode(y)
        return y

