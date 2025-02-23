import argparse
import os
from util import util
import torch
import duplex_model
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', required=True, help='path to images folder')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='/nrs/funke/senetaire/checkpoints/', help='models are saved here')
        parser.add_argument('--isTrain', action='store_true', help='train or test')
        # model parameters
        parser.add_argument('--model', type=str, default='pathwise_selector', help='chooses which model to use. [pathwise_selector| pathwise_selector_otf | pathwise_selector_pair_dic | test]')

        # selector parameters
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--net_selector', type=str, default='resnet_6blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128| asymmetric_unet_128 | unet_32, fc]')
        parser.add_argument('--downscale_asymmetric', type=int, default=1, help='if specified and use asymmetric unet, the produced mask will be downsampled by this 2**downscale_asymmetric factor')
        parser.add_argument('--upscale_after_sampling', action='store_true', help='if specified and use asymmetric unet, the produced mask will be upsampled by this 2**downscale_asymmetric factor but after samplin\
                            otherwise pi will be upsampled by this 2**downscale_asymmetric factor, doesnt change a thing for pi_as_mask') # TODO : Implement this for later ? Not sure if really required
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')

        # classifier parameters
        parser.add_argument('--f_theta_checkpoint', type=str, default=None, help='Path to classifier checkpoint to restore weights from')
        parser.add_argument('--f_theta_input_shape', type=int, nargs='+', default=[128, 128], help='Input shape for classifier')
        parser.add_argument('--f_theta_input_nc', type=int, default=1, help='Input channels for classifier')
        parser.add_argument('--f_theta_net', type=str, default='Vgg2D', help='Name of classifier')
        parser.add_argument('--f_theta_output_classes', type=int, default=6, help='Number of output classes for classifier')


        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='synapsefolder', help='chooses how datasets are loaded. [mnistduck, synapsefolder, synapsenocf]')
        parser.add_argument('--no_augment', action='store_true', help='if specified, do not augment the data')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=128, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=128, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='none', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_false', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{net_selector}_size{load_size}')
        self.initialized = True
        return parser

    def gather_options(self, input = None):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)


        # get the basic options
        opt, _ = parser.parse_known_args(input)

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = duplex_model.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args(input)  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args(input)

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, input = None):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options(input = input)
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
