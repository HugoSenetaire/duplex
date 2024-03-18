from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """



    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # trainer parameters
        parser.add_argument('--trainer', type=str, default='pathwise', choices = ['pathwise', 'pathwise_otf', 'pathwise_pairdic', 'test'],
                            help='chooses which trainer to use. [pathwise| pathwise_otf | pathwise_pairdic | test]')

        parser.add_argument('--n_epochs_pretrain_deeplift', type=int, default=0, help='Number of epochs to pretrain the deeplift model')




        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=500, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        parser.add_argument('--log_freq', type=int, default=100, help='frequency of saving training results to log')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=200, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        # Training parameters specific
        parser.add_argument('--mc_sample_z', type=int, default=1, help='Number of MC samples for z')
        parser.add_argument('--imp_sample_z', type=int, default=1, help='Number of Importance samples for z')
        parser.add_argument('--mc_sample_tilde_x', type=int, default=1, help='Number of MC samples for tilde_x')
        parser.add_argument('--imp_sample_tilde_x', type=int, default=1, help='Number of Importance samples for tilde_x')

        parser.add_argument('--mc_sample_z_test', type=int, default=1, help='Number of MC samples for z')
        parser.add_argument('--imp_sample_z_test', type=int, default=1, help='Number of Importance samples for z')
        
        self.isTrain = True
        return parser
