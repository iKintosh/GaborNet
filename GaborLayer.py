import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.conv import Conv2d


class Gabor(Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.sigma_x = nn.Parameter(torch.rand(in_channels*out_channels))
        self.sigma_y = nn.Parameter(torch.rand(in_channels*out_channels))
        self.freq = nn.Parameter(torch.rand(in_channels*out_channels))
        self.theta = nn.Parameter(torch.rand(in_channels*out_channels))
        self.psi = nn.Parameter(torch.rand(in_channels*out_channels))
        #self.evaluate = False

    # def eval(self):
        #self.evaluate = True

    # def train(self):
        #self.evaluate = False

    def forward(self, input):
        '''if self.evaluate == True:
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)'''

        x0 = torch.ceil(torch.Tensor([self.kernel_size[0]/2]))[0]
        y0 = torch.ceil(torch.Tensor([self.kernel_size[1]/2]))[0]

        weight = torch.empty(self.weight.shape, requires_grad=False)
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                y, x = torch.meshgrid([torch.arange(-y0+1, y0), torch.arange(-x0+1, x0)])

                sigma_x = self.sigma_x[i*j].expand_as(y)
                sigma_y = self.sigma_y[i*j].expand_as(y)
                freq = self.freq[i*j].expand_as(y)
                theta = self.theta[i*j].expand_as(y)
                psi = self.psi[i*j].expand_as(y)

                rotx = x * torch.cos(theta) + y * torch.sin(theta)
                roty = -x * torch.sin(theta) + y * torch.cos(theta)
                g = torch.zeros(y.shape)
                g = torch.exp(-0.5 * (rotx ** 2 / (sigma_x + 1e-3) ** 2 + roty ** 2 / (sigma_y + 1e-3) ** 2))
                g = g * torch.cos(2 * 3.14 * freq * rotx + psi)
                weight[j, i] = g
                self.weight.data[j, i] = g
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
