import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from dapi_networks.Vgg2D import Vgg2D
import numpy as np
from dapi_networks.ResNet import ResNet

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer




def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], checkpoint_path = None):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    
    if checkpoint_path is not None:
        net.load_state_dict(torch.load(checkpoint_path))
    else :
        init_weights(net, init_type, init_gain=init_gain)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    
    return net


def define_selector(   
        input_nc,
        ngf,
        selector,
        norm='batch',
        use_dropout=False,
        init_type='normal',
        init_gain=0.02,
        gpu_ids=[],
        input_shape=(32,32),
        downscale_asymmetric=1,
        checkpoint_selector = None,
        ):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images        
        ngf (int) -- the number of filters in the last conv layer
        selector (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128 
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        downscale_asymmetric (int) -- if using asymmetric unet, how much to downscale by
                                    if 1, we will have superpizels of size2x2,
                                    if 2, we will have superpixels of size 4x4, etc.
        checkpoint_selector (str) -- path to a checkpoint to load the selector from

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    output_nc = 1 # Mask output so single channel

    if "asymmetric" in selector:
        assert downscale_asymmetric > 0 and downscale_asymmetric < 7, "downscale_asymmetric should be between 1 and 6 if using asymmetric unet"
    else :
        assert downscale_asymmetric == 0, "downscale_asymmetric should be 0 if not using asymmetric unet"
    if selector == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif selector == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif selector == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif selector== 'asymmetric_unet_224':
        net = AsymmetricUNetGeneratorRobust(input_nc, output_nc, 7, 7-downscale_asymmetric, ngf, norm_layer=norm_layer, use_dropout=use_dropout,)
    elif selector=='asymmetric_unet_128':
        net = AsymmetricUNetGenerator(input_nc, output_nc, 7, 7-downscale_asymmetric, ngf, norm_layer=norm_layer, use_dropout=use_dropout, )
    elif selector == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif selector == 'unet_32':
        net = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif selector == 'fc':
        net = FullyConectedGenerator(input_shape)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % selector)
    return init_net(net, init_type, init_gain, gpu_ids, checkpoint_path=checkpoint_selector)




##############################################################################
# Classes
##############################################################################



class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class FullyConectedGenerator(nn.Module):
    """Fully connected generator"""
    def __init__(self, input_size,):
        super(FullyConectedGenerator, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, np.prod(input_size))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):   
        x = x.view(-1, np.prod(self.input_size))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = x.view(-1, 1, self.input_size[0], self.input_size[1])
        return x


        
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv,]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UnetGeneratorv2(nn.Module):
    """Create a Unet-based generator specifically to handle a case of 224x224 size"""
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGeneratorv2, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 2, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetGeneratorRobust(nn.Module):
    """Create a Unet-based generator that allows even for non power of 2 input sizes"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example,
                                if |num_downs| == 7, image of size 128x128 will become of size 1x1 at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(UnetGeneratorRobust, self).__init__()
        nb_blocks = num_downs - 2
        
        i=0
        max_ngf_outer = min(ngf * 2 ** (nb_blocks - 2 -i), 2**8)
        max_ngf_inner = min(ngf * 2 ** (nb_blocks -1 - i), 2**8)
        unet_block = UnetSkipConnectionBlockRobust(max_ngf_outer, max_ngf_outer, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(1, nb_blocks-1):
            max_ngf_outer = min(ngf * 2 ** (nb_blocks - 2 -i), 2**8)
            max_ngf_inner = min(ngf * 2 ** (nb_blocks -1 - i), 2**8)
            unet_block = UnetSkipConnectionBlockRobust(max_ngf_outer, max_ngf_inner, input_nc=None, submodule=unet_block, norm_layer=norm_layer)

        assert max_ngf_outer == ngf
    
        # gradually reduce the number of filters from ngf * 8 to ngf
        self.model = UnetSkipConnectionBlockRobust(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer



    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlockRobust(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlockRobust) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlockRobust, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv,]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            skip = x
            gen = self.model(x)[:,:, :skip.size(2), :skip.size(3)]
            return torch.cat([skip, gen], 1)
            

class UpBlock(nn.Module):
    """Up-sampling block using learnable convolution"""
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        """Construct an up-sampling block

        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            norm_layer      -- normalization layer
        """
        super(UpBlock, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else: 
            use_bias = norm_layer == nn.InstanceNorm2d
        
        self.upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
        self.conv = nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.uprelu = nn.ReLU(True)
        self.upnorm = norm_layer(output_nc)

    def forward(self, x):
        x = self.uprelu(self.upconv(x))
        x = self.upnorm(self.conv(x))
        return x

class UpSampleBlock(nn.Module):
    """Up-sampling block without any learned convolution"""
    def __init__(self, ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        return self.upsample(x)


class DownBlock(nn.Module):
    """Down-sampling block"""
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        """Construct a down-sampling block

        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            norm_layer      -- normalization layer
        """
        super(DownBlock, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        self.conv1 = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        # self.conv2 = nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.downrelu = nn.LeakyReLU(0.2)
        self.downnorm = norm_layer(output_nc)

    def forward(self, x):
        x = self.downrelu(self.conv1(x))
        x = self.downnorm(x) 
        return x


class AsymmetricUNetGenerator(nn.Module):
    """Create a UNet-based generator with a different number of downsampling block and upsampling blocks"""

    def __init__(self, input_nc, output_nc, num_downs, num_ups, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, upscale_after_sampling=False, use_v2=False ):
        """Construct the asymmetric UNet generator
        
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet
            num_ups (int)   -- the number of upsamplings in UNet
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        # TODO make this asymmetric both ways
        super(AsymmetricUNetGenerator, self).__init__()
        # assert num_downs >= num_ups

        num_skips = min(num_downs, num_ups)
        num_downsample = num_downs - num_skips
        num_upsample = num_ups - num_skips
        print("Num Skip: ", num_skips)
        print("Num Down: ", num_downsample)
        print("Num Up: ", num_upsample)

        pre_base = []
        post_base = []
        unet_input = input_nc
        unet_output = output_nc

        # Potential skip-less downsampling 
        # We need to downsample more than we upsample
        
        for i in range(num_downsample):
            pre_base += [DownBlock(input_nc, ngf * 2**(i), norm_layer)]
            input_nc = ngf
            ngf = ngf * 2**(i)
        unet_input = ngf


        # Potential skip-less upsampling, potential super resolution
        # We need to upsample more than we downsample
        for i in range(num_upsample):
            if i == 0: 
                unet_output = ngf
            if i == num_ups - num_downs - 1:
                # Get the right number of final channels
                post_base += [UpBlock(ngf, output_nc, norm_layer)]
            else:
                post_base += [UpBlock(ngf, ngf // 2, norm_layer)]
            ngf = ngf // 2



        self.down_sampling = nn.Sequential(*pre_base)
        if use_v2 :
            self.base_model = UnetGeneratorv2(unet_input, unet_output, num_downs=num_skips, ngf=ngf)
        else :
            self.base_model = UnetGenerator(unet_input, unet_output, num_downs=num_skips, )
        self.up_sampling = nn.Sequential(*post_base)

    def forward(self, x):
        x = self.down_sampling(x)
        x = self.base_model(x)
        x = self.up_sampling(x)
        return x




class AsymmetricUNetGeneratorRobust(nn.Module):
    """Create a UNet-based generator with a different number of downsampling block and upsampling blocks"""

    def __init__(self, input_nc, output_nc, num_downs, num_ups, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, upscale_after_sampling=False,):
        """Construct the asymmetric UNet generator
        
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet
            num_ups (int)   -- the number of upsamplings in UNet
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        # TODO make this asymmetric both ways
        super(AsymmetricUNetGeneratorRobust, self).__init__()
        # assert num_downs >= num_ups

        num_skips_connections = min(num_downs, num_ups)
        num_downsample = num_downs - num_skips_connections
        num_upsample = num_ups - num_skips_connections
        print("Num Skip: ", num_skips_connections)
        print("Num Down: ", num_downsample)
        print("Num Up: ", num_upsample)

        pre_base = []
        post_base = []
        unet_input = input_nc
        unet_output = output_nc

        # Potential skip-less downsampling 
        # We need to downsample more than we upsample
        
        for i in range(num_downsample):
            pre_base += [DownBlock(input_nc, ngf * 2**(i), norm_layer)]
            input_nc = ngf
            ngf = ngf * 2**(i)
        unet_input = ngf


        # Potential skip-less upsampling, potential super resolution
        # We need to upsample more than we downsample
        for i in range(num_upsample):
            if i == 0: 
                unet_output = ngf
            if i == num_ups - num_downs - 1:
                # Get the right number of final channels
                post_base += [UpBlock(ngf, output_nc, norm_layer)]
            else:
                post_base += [UpBlock(ngf, ngf // 2, norm_layer)]
            ngf = ngf // 2



        self.down_sampling = nn.Sequential(*pre_base)
        self.base_model = UnetGeneratorRobust(unet_input, unet_output, num_downs=num_skips_connections, ngf=ngf)
        self.up_sampling = nn.Sequential(*post_base)

    def forward(self, x):
        x = self.down_sampling(x)
        x = self.base_model(x)
        x = self.up_sampling(x)
        return x