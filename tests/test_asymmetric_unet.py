import pytest
import torch
from selector.duplex_model.networks import AsymmetricUNetGenerator, DownBlock, UpBlock, UnetGenerator
from torchinfo import summary

@pytest.mark.parametrize("input_nc, output_nc", [(1, 4), (4, 8)])
def test_downblock(input_nc, output_nc):
    input_size = (1, input_nc, 256, 256)
    model = DownBlock(input_nc=input_nc, output_nc=output_nc)
    x = model(torch.randn(input_size))
    assert x.shape == (1, output_nc, 128, 128)


@pytest.mark.parametrize("input_nc, output_nc", [(8, 4), (4, 1)])
def test_upblock(input_nc, output_nc):
    input_size = (1, input_nc, 128, 128)
    model = UpBlock(input_nc=input_nc, output_nc=output_nc)
    x = model(torch.randn(input_size))
    assert x.shape == (1, output_nc, 256, 256)


@pytest.mark.parametrize("ngf, num_downs", [(64, 2)]) #, (3, 3, 4)])
def test_unet_generator(ngf, num_downs):
    input_size = (1, ngf, 128, 128)
    model = UnetGenerator(ngf, ngf, num_downs=num_downs)
    x = model(torch.randn(input_size))
    out = model(x)
    assert out.shape == (1, ngf, 128, 128)


@pytest.mark.parametrize("num_downs, num_ups", [(3, 2), (4, 2), (2, 4), (4, 4)]) #, 4])
def test_asymmetric_unet(num_downs, num_ups):
    input_size = (1, 1, 256, 256)
    model = AsymmetricUNetGenerator(input_nc=1, output_nc=1, num_downs=num_downs, num_ups=num_ups)
    x = torch.randn(input_size)
    out = model(x)
    output_size = (1, 1, 256 / 2 ** (num_downs - num_ups), 256 / 2 ** (num_downs - num_ups)) 
    assert out.shape == output_size