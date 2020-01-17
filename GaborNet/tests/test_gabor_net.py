import numpy as np
import torch

from GaborNet import GaborConv2d


def test_kenrel_shape():
    k = np.random.randint(11, 15)
    layer = GaborConv2d(1, 1, (k, k))
    shape = np.array(layer.weight.shape)
    result = k
    assert result == shape[-1]


def test_output_size():
    s = np.random.randint(3, 9)
    layer = GaborConv2d(1, s, (11, 11))
    shape = np.array(layer.weight.shape)
    result = s
    assert result == shape[0]


def test_input_size():
    s = np.random.randint(3, 9)
    layer = GaborConv2d(s, 1, (11, 11))
    shape = np.array(layer.weight.shape)
    result = s
    assert result == shape[1]


def test_layer_work():
    img = np.random.randn(1, 1, 100, 100)
    img = torch.Tensor(img)
    k = np.random.randint(11, 15)
    layer = GaborConv2d(1, 1, (k, k))
    output = layer(img)
    shape = np.array(output.shape)
    result = 100 - k + 1
    assert result == shape[-1] == shape[-2]
