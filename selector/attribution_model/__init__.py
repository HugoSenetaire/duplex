import torch

from .networks import define_selector, get_norm_layer
from .mask_distribution import get_mask_distribution
from .duplex_model import DupLEX
from dapi_networks.network_utils import init_network



def find_mask_distribution_using_name(mask_distribution_name):
    """Import the module "duplex_mask_distribution/[mask_distribution_name]_mask_distribution.py".

    In the file, the class called DatasetNamemask_distribution() will
    be instantiated. It has to be a subclass of Basemask_distribution,
    and it is case-insensitive.
    """
    # mask_distribution_filename = "attribution_model." + mask_distribution_name + "_mask_distribution"
    # mask_distributionlib = importlib.import_module(mask_distribution_filename)
    # mask_distribution = None
    # target_mask_distribution_name = mask_distribution_name.replace('_', '') + 'mask_distribution'
    # for name, cls in mask_distributionlib.__dict__.items():
    #     if name.lower() == target_mask_distribution_name.lower() \
    #        and issubclass(cls, AbstractMaskDistribution):
    #         mask_distribution = cls

    # if mask_distribution is None:
    #     print("In %s.py, there should be a subclass of Basemask_distribution with class name that matches %s in lowercase." % (mask_distribution_filename, target_mask_distribution_name))
    #     exit(0)

    current_mask_distribution= get_mask_distribution(mask_distribution_name)

    return current_mask_distribution


def get_option_setter(mask_distribution_name):
    """Return the static method <modify_commandline_options> of the mask_distribution class."""
    mask_distribution_class = find_mask_distribution_using_name(mask_distribution_name)
    return mask_distribution_class.modify_commandline_options


def create_mask_distribution(opt):
    """Create a mask_distribution given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from mask_distributions import create_mask_distribution
        >>> mask_distribution = create_mask_distribution(opt)
    """
    mask_distribution = find_mask_distribution_using_name(opt.mask_distribution)
    instance = mask_distribution(opt)
    print("mask_distribution [%s] was created" % type(instance).__name__)
    return instance


def initAttributionModel(opt,):
    """Create a duplex model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from duplex_model import initAttributionModel
        >>> duplex_model = initAttributionModel(opt)
    """
    classifier = init_network(
            opt.f_theta_checkpoint,
            opt.f_theta_input_shape,
            opt.f_theta_net,
            opt.f_theta_input_nc,
            opt.f_theta_output_classes,
            downsample_factors=[(2, 2), (2, 2), (2, 2), (2, 2)]
            )

    if opt.use_counterfactual_as_input:
        opt.input_nc *= 2
    
    selector = define_selector(
            opt.input_nc,
            opt.ngf,
            opt.net_selector,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            opt.gpu_ids,
            opt.f_theta_input_shape,
            downscale_asymmetric=opt.downscale_asymmetric,
            )
    
    # If generated mask requires upscaling
    upscale = ("asymmetric" in opt.net_selector and opt.downscale_asymmetric > 0)
    if upscale :
        upscaler = torch.nn.Upsample(scale_factor=2**opt.downscale_asymmetric, mode='nearest')
    else :
        upscaler = None
    mask_distribution = create_mask_distribution(opt)

    duplex_model = DupLEX(
                        classifier,
                        selector,
                        mask_distribution,
                        upscaler=upscaler,
                        use_counterfactual_as_input=opt.use_counterfactual_as_input,
                        param_gaussian_smoothing_sigma=opt.param_gaussian_smoothing_sigma,
                        )

    return duplex_model