# GaborNet

[![PyPI-Status][pypi-image]][pypi-url]
[![Build Status][travis-badge]][travis-url]
[![LICENSE][license-image]][license-url]
[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/iKintosh/GaborNet/?ref=repository-badge)

## Installation

GaborNet can be installed via pip from Python 3.7 and above:

```bash
pip install GaborNet
```

## Getting started

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
from GaborNet import GaborConv2d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GaborNN(nn.Module):
    def __init__(self):
        super(GaborNN, self).__init__()
        self.g0 = GaborConv2d(in_channels=1, out_channels=96, kernel_size=(11, 11))
        self.c1 = nn.Conv2d(96, 384, (3,3))
        self.fc1 = nn.Linear(384*3*3, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.leaky_relu(self.g0(x))
        x = nn.MaxPool2d()(x)
        x = F.leaky_relu(self.c1(x))
        x = nn.MaxPool2d()(x)
        x = x.view(-1, 384*3*3)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = GaborNN().to(device)

```

Original research paper (preprint): https://arxiv.org/abs/1904.13204

This research on deep convolutional neural networks proposes a modified
architecture that focuses on improving convergence and reducing training
complexity. The filters in the first layer of network are constrained to fit the
Gabor function. The parameters of Gabor functions are learnable and updated by
standard backpropagation techniques. The proposed architecture was tested on
several datasets and outperformed the common convolutional networks

## Citation

Please use this bibtex if you want to cite this repository in your publications:

    @misc{gabornet,
        author = {Alekseev, Andrey},
        title = {GaborNet: Gabor filters with learnable parameters in deep convolutional
                   neural networks},
        year = {2019},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/iKintosh/GaborNet}},
    }

[travis-url]: https://travis-ci.com/iKintosh/GaborNet
[travis-badge]: https://travis-ci.com/iKintosh/GaborNet.svg?branch=master
[pypi-image]: https://img.shields.io/pypi/v/gabornet.svg
[pypi-url]: https://pypi.org/project/gabornet
[license-image]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://pypi.org/project/gabornet
