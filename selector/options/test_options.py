from .base_options import BaseOptions

def downsample_type(s):
    try:
        factors = [tuple([int(x) for x in l.split(',')]) for l in s.split('x')]
    except:
        raise ValueError("Downsample input not understood")

    return factors


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options

        parser.add_argument('--results_dir', type=str, default=None, help='saves results here. If none, will save to checkpoints_dir/name/results')
        parser.add_argument('--load_epoch', type=int, default=100, help='checkpoint number to load')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        
        # Test parameters specific
        parser.add_argument('--mc_sample_z_test', type=int, default=1, help='Number of MC samples for z')
        parser.add_argument('--imp_sample_z_test', type=int, default=1, help='Number of Importance samples for z')

        # rewrite devalue values
        # parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        # parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
