import torch
import torch.nn as nn
import torch.nn.functional as F
from GaborLayers import GaborConv2d

DOWNSAMPLE_COEF = 2
BOTTLENECK_COEF = 4


def conv3x3(a_in_planes, a_out_planes, a_stride=1):
    """
    Основной строительный блок конволюций для ResNet
    Включает в себя padding=1 - чтобы размерность сохранялась после его применения
    """
    return nn.Conv2d(a_in_planes, a_out_planes,  stride=a_stride,
                     kernel_size=3, padding=1, bias=False)


def gabor15x15(a_in_planes, a_out_planes, a_stride=1):
    """
    Основной строительный блок конволюций для GaborResNet
    Включает в себя padding=7 - чтобы размерность сохранялась после его применения
    """
    return GaborConv2d(a_in_planes, a_out_planes,  stride=a_stride,
                       kernel_size=15, padding=7, bias=False)


def x_downsample(a_in_channels):
    return nn.Conv2d(a_in_channels,
                     a_in_channels * DOWNSAMPLE_COEF,
                     kernel_size=1,
                     stride=DOWNSAMPLE_COEF,
                     bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, a_in_channels, make_downsample=False, use_skip_connection=True):
        super(ResidualBlock, self).__init__()
        self.use_skip_connection = use_skip_connection
        self.make_downsample = make_downsample

        if make_downsample:
            coef = DOWNSAMPLE_COEF
        else:
            coef = 1

        self.ReLU = nn.ReLU(inplace=True)
        self.BatchNorm1 = nn.BatchNorm2d(a_in_channels*coef)
        self.BatchNorm2 = nn.BatchNorm2d(a_in_channels*coef)
        self.conv1st = conv3x3(a_in_channels, a_in_channels*coef, a_stride=coef)
        self.conv2st = conv3x3(a_in_channels*coef, a_in_channels*coef)

        if make_downsample:
            self.downsample = x_downsample(a_in_channels)
        else:
            self.downsample = None

    def forward(self, x):
        tmp = self.conv1st(x)
        tmp = self.BatchNorm1(tmp)
        tmp = self.ReLU(tmp)

        tmp = self.conv2st(tmp)
        tmp = self.BatchNorm2(tmp)

        if self.use_skip_connection:
            if self.make_downsample:
                x = self.downsample(x)
            tmp += x
        tmp = self.ReLU(tmp)

        return tmp


class GaborResidualBlock(nn.Module):
    def __init__(self, a_in_channels, make_downsample=False, use_skip_connection=True):
        super(GaborResidualBlock, self).__init__()
        self.use_skip_connection = use_skip_connection
        self.make_downsample = make_downsample

        if make_downsample:
            coef = DOWNSAMPLE_COEF
        else:
            coef = 1

        self.ReLU = nn.ReLU(inplace=True)
        self.BatchNorm1 = nn.BatchNorm2d(a_in_channels*coef)
        self.BatchNorm2 = nn.BatchNorm2d(a_in_channels*coef)
        self.conv1st = gabor15x15(a_in_channels, a_in_channels*coef, a_stride=coef)
        self.conv2st = gabor15x15(a_in_channels*coef, a_in_channels*coef)

        if make_downsample:
            self.downsample = x_downsample(a_in_channels)
        else:
            self.downsample = None

    def forward(self, x):
        tmp = self.conv1st(x)
        tmp = self.BatchNorm1(tmp)
        tmp = self.ReLU(tmp)

        tmp = self.conv2st(tmp)
        tmp = self.BatchNorm2(tmp)

        if self.use_skip_connection:
            if self.make_downsample:
                x = self.downsample(x)
            tmp += x
        tmp = self.ReLU(tmp)

        return tmp


class ResidualBottleneckBlock(nn.Module):

    def __init__(self, a_in_channels, make_downsample=False, use_skip_connection=True):
        super(ResidualBottleneckBlock, self).__init__()
        self.use_skip_connection = use_skip_connection
        self.use_skip_connection = use_skip_connection
        self.make_downsample = make_downsample

        if make_downsample:
            coef = DOWNSAMPLE_COEF
        else:
            coef = 1

        self.ReLU = nn.ReLU(inplace=True)
        self.BatchNorm1 = nn.BatchNorm2d(a_in_channels*coef//BOTTLENECK_COEF)
        self.BatchNorm2 = nn.BatchNorm2d(a_in_channels*coef//BOTTLENECK_COEF)
        self.BatchNorm3 = nn.BatchNorm2d(a_in_channels*coef)

        self.preconv = nn.Conv2d(a_in_channels,
                                 a_in_channels*coef//BOTTLENECK_COEF,
                                 kernel_size=1,
                                 stride=1,
                                 bias=False)
        self.conv1st = conv3x3(a_in_channels*coef//BOTTLENECK_COEF,
                               a_in_channels*coef//BOTTLENECK_COEF, a_stride=coef)
        self.conv2nd = nn.Conv2d(a_in_channels*coef//BOTTLENECK_COEF,
                                 a_in_channels*coef,
                                 kernel_size=1,
                                 stride=1,
                                 bias=False)

        if make_downsample:
            self.downsample = x_downsample(a_in_channels)
        else:
            self.downsample = None

    def forward(self, x):
        res = x

        tmp = self.preconv(x)
        tmp = self.BatchNorm1(tmp)
        tmp = self.ReLU(tmp)

        tmp = self.conv1st(tmp)
        tmp = self.BatchNorm2(tmp)
        tmp = self.ReLU(tmp)

        tmp = self.conv2nd(tmp)
        tmp = self.BatchNorm3(tmp)

        if self.make_downsample:
            res = self.downsample(x)
        if self.use_skip_connection:
            tmp += res
        tmp = self.ReLU(tmp)

        return tmp


class GaborResidualBottleneckBlock(nn.Module):

    def __init__(self, a_in_channels, make_downsample=False, use_skip_connection=True):
        super(GaborResidualBottleneckBlock, self).__init__()
        self.use_skip_connection = use_skip_connection
        self.use_skip_connection = use_skip_connection
        self.make_downsample = make_downsample

        if make_downsample:
            coef = DOWNSAMPLE_COEF
        else:
            coef = 1

        self.ReLU = nn.ReLU(inplace=True)
        self.BatchNorm1 = nn.BatchNorm2d(a_in_channels*coef//BOTTLENECK_COEF)
        self.BatchNorm2 = nn.BatchNorm2d(a_in_channels*coef//BOTTLENECK_COEF)
        self.BatchNorm3 = nn.BatchNorm2d(a_in_channels*coef)

        self.preconv = nn.Conv2d(a_in_channels,
                                 a_in_channels*coef//BOTTLENECK_COEF,
                                 kernel_size=1,
                                 stride=1,
                                 bias=False)
        self.conv1st = gabor15x15(a_in_channels*coef//BOTTLENECK_COEF,
                                  a_in_channels*coef//BOTTLENECK_COEF, a_stride=coef)
        self.conv2nd = nn.Conv2d(a_in_channels*coef//BOTTLENECK_COEF,
                                 a_in_channels*coef,
                                 kernel_size=1,
                                 stride=1,
                                 bias=False)

        if make_downsample:
            self.downsample = x_downsample(a_in_channels)
        else:
            self.downsample = None

    def forward(self, x):
        res = x

        tmp = self.preconv(x)
        tmp = self.BatchNorm1(tmp)
        tmp = self.ReLU(tmp)

        tmp = self.conv1st(tmp)
        tmp = self.BatchNorm2(tmp)
        tmp = self.ReLU(tmp)

        tmp = self.conv2nd(tmp)
        tmp = self.BatchNorm3(tmp)

        if self.make_downsample:
            res = self.downsample(x)
        if self.use_skip_connection:
            tmp += res
        tmp = self.ReLU(tmp)

        return tmp
