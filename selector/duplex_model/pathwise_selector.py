import torch
import torch.nn.functional as F
import itertools
from util.image_pool import ImagePool
from duplex_model.base_model import BaseModel
from duplex_model.mask_distribution import IndependentRelaxedBernoulli
from duplex_model.networks import define_selector
from classifier_networks.network_utils import init_network, run_inference


class PathWiseSelector(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        parser.add_argument('--lambda_regularization', type=float, default=0.1, help='L1 regularization strenght to limit the selection of the mask')
        parser.add_argument('--temperature_relax', type=float, default=0.1, help='Temperature for the relaxed mask distribution')

        return parser

    def __init__(self, opt):
        """Initialize the Selector class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["loss_class", "loss_reg"]

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['x', 'x_cf', 'z', 'x_tilde',]

        # check args consistency
        # NA for Now

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['g_gamma', 'f_theta',]
        else:  # during test time, only load Gs
            self.model_names = ['g_gamma', ]

        # define networks (both selectors and classifiers)
        self.netg_gamma = define_selector(opt.input_nc, opt.ngf, opt.net_selector, opt.norm, # In the case of the mask, one just needs the output mask
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        
        self.p_z = IndependentRelaxedBernoulli(temperature_relax=opt.temperature_relax)  # mask distribution

        if self.isTrain:  # define classifiers
            self.netf_theta = init_network(
                checkpoint_path=opt.f_theta_checkpoint,
                input_shape=opt.input_shape,
                net_module=opt.f_theta_net,
                input_nc=opt.f_theta_input_nc,
                output_classes=opt.f_theta_output_classes,
                )
            

        if self.isTrain:

            assert opt.lambda_regularization > 0.0, "lambda_regularization must be > 0.0"
            self.lambda_regularization = opt.lambda_regularization

            assert self.p_z.rsample_available, "rsample must be available for the mask distribution in order to use pathwise gradient estimator"
            
            

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_selector = torch.optim.Adam(self.selector.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_selector)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.x = input['x'].to(self.device)
        self.y = input['y_true'].to(self.device)

        self.x_cf = input['x_cf'].to(self.device) # TODO: @hhjs Multiple cp here ?
        self.y_cf = input['y_cf'].to(self.device) 
        self.image_paths = input['path']

    def define_nb_sample(self,):
        """
        Define the number of samples to draw from the mask distribution.
        Depending on the training/testing phase, the number of samples is different.
        Note that the evaluation of the likelihood is dependent on the number of mask samples.
        """
        if self.netg_gamma.training:
            self.mc_sample_z = self.opt.mc_sample_z
            self.imp_sample_z = self.opt.imp_sample_z
        else:
            self.mc_sample_z = self.opt.mc_sample_z_test
            self.imp_sample_z = self.opt.imp_sample_z_test
    

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.define_nb_sample()
        
        # Calculate the mask distribution parameter
        self.pi = self.netg_gamma(self.x)

        # Sample from the mask distribution
        self.z = self.p_z.rsample(self.pi, self.mc_sample_z) # Need Rsample here to allow pathwise estimation
        self.z = self.z.reshape(self.mc_sample_z, self.x.shape[0], *self.x.shape[2:]) 
        
        # Expand the input images to match the shape of the mask samples
        self.x_expanded = self.x.unsqueeze(0).expand(self.mc_sample_z,*self.x.shape)
        self.x_cf_expanded = self.x_cf.unsqueeze(1).expand(self.mc_sample_z, *self.x_cf.shape)
        
        # Create mixed images
        self.x_tilde = (self.x_expanded * self.z + (1 - self.z) * self.x_cf_expanded).flatten(0,1)
        
        # Calculate the classifier output on the mixed images
        self.y_tilde = self.netf_theta(self.x_tilde).reshape(self.mc_sample_z, self.x.shape[0])
        self.y_expanded = self.y.unsqueeze(0).expand(self.mc_sample_z, *self.y.shape)


    def backward_g_gamma(self):
        """Calculate the loss selector NN g_gamma"""
        # Regularization
        self.loss_reg = self.lambda_regularization * self.z.reshape(self.x.shape[0], self.mc_sample_z*self.imp_sample_z, *self.x.shape[1:]).abs().mean(1).mean(0).mean()
        
        # Likelihood guidance 
        self.loss_class = F.cross_entropy(self.y_tilde, self.y_expanded)

        self.loss_total = self.loss_class + self.loss_reg
        self.loss_total.backward()
       


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # Compute mask and mixed images
        self.set_requires_grad([self.netg_gamma,], True)
        self.optimizer_selector.zero_grad()  # set g_gamma's gradients to zero
        self.backward_g_gamma()             # calculate gradients g_gamma
        self.optimizer_selector.step()       # update g_gamma's weights
       
