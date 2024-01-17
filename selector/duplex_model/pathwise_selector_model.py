import torch
import torch.nn.functional as F
import itertools
from duplex_model.base_selector import BaseSelector
from duplex_model.mask_distribution import IndependentRelaxedBernoulli
from duplex_model.networks import define_selector
from dapi_networks.network_utils import init_network, run_inference
from util.gaussian_smoothing import gaussian_filter_2d
from duplex_model.scheduler_lambda import get_scheduler_lambda


class PathWiseSelectorModel(BaseSelector):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True, opt = None):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        parser.add_argument('--use_pi_as_mask', action='store_true', help='If specified, the mask distribution is not sampled and we directly optimize on pi. \
                                                                        This is only possible when the imputation method is deterministic (or very simple).')
        parser.add_argument('--pi_gaussian_smoothing_sigma', type=float, default=-1.0, help='If specified, the pis mask \
                                                        is smoothed with a gaussian filter of sigma gaussian_smoothing_sigma')
        parser.add_argument('--z_gaussian_smoothing_sigma', type=float, default=-1.0, help='If specified, the sampled mask \
        is smoothed with a gaussian filter of sigma gaussian_smoothing_sigma. In the case of use_pi_as_mask, it is still different \
        from pi_gaussian_smoothing_sigma  as the smoothing happens after the upsampling.')
        

        parser.add_argument('--lambda_regularization', type=float, default=1.0, help='L1 regularization strenght to limit the selection of the mask')
        parser.add_argument('--lambda_regularization_init', type=float, default=0., help='Initial value for the lambda_regularization scheduler')
        parser.add_argument('--lambda_regularization_scheduler', type=str, default='constant', help='Scheduler for the lambda_regularization parameter. [linear | constant | cosine]')
        parser.add_argument('--lambda_regularization_scheduler_targetepoch', type=int, default=100, help='Target epoch for the lambda_regularization scheduler to reach the lambda_regularization value')


        parser.add_argument('--temperature_relax', type=float, default=1.0, help='Temperature for the relaxed mask distribution')

        parser.add_argument('--lambda_ising_regularization', type=float, default=-1.0, help='Ising regularization strenght to enforce connectivity in the mask selection')
        parser.add_argument('--lambda_ising_regularization_init', type=float, default=0., help='Initial value for the lambda_ising_regularization scheduler')
        parser.add_argument('--lambda_ising_regularization_scheduler', type=str, default='constant', help='Scheduler for the lambda_ising_regularization parameter. [linear | constant | cosine]')
        parser.add_argument('--lambda_ising_regularization_scheduler_targetepoch', type=int, default=100, help='Target epoch for the lambda_ising_regularization scheduler to reach the lambda_ising_regularization value')
        
        return parser

    def __init__(self, opt):
        """Initialize the Selector class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseSelector.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["ising_regularization", "reg", "class_z", "class_notemp", "class_no_selector", "class_pi", "acc_z", "acc_notemp", "acc_no_selector", "acc_pi", "quantile_pi_25", "quantile_pi_50", "quantile_pi_75"]

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['x', 'x_cf', 'pi_to_save', 'x_tilde_pi', 'z_to_save', 'x_tilde_z',  'z_to_save_notemp', 'x_tilde_notemp']

        self.use_pi_as_mask = opt.use_pi_as_mask
        self.pi_gaussian_smoothing_sigma = opt.pi_gaussian_smoothing_sigma
        self.z_gaussian_smoothing_sigma = opt.z_gaussian_smoothing_sigma
        self.lambda_ising_regularization = opt.lambda_ising_regularization


        # If generated mask requires upscaling
        self.upscale = ("asymmetric" in opt.net_selector and opt.downscale_asymmetric > 0)
        if self.upscale :
            self.upscaler = torch.nn.Upsample(scale_factor=2**opt.downscale_asymmetric, mode='nearest')
        self.upscale_after_sampling = opt.upscale_after_sampling
        # check args consistency
        # NA for Now

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['g_gamma', 'f_theta',]
        else:  # during test time, only load Gs
            self.model_names = ['g_gamma', ]

        # define networks (both selectors and classifiers)
        print("Setting up selector")
        self.netg_gamma = define_selector(
                                opt.input_nc,
                                opt.ngf,
                                opt.net_selector,
                                opt.norm, # In the case of the mask one just needs the output mask
                                not opt.no_dropout,
                                opt.init_type,
                                opt.init_gain,                            
                                self.gpu_ids,
                                opt.f_theta_input_shape,
                                downscale_asymmetric=opt.downscale_asymmetric,
                                ).to(self.device)
        
        print("Setting up mask distribution")
        self.p_z = IndependentRelaxedBernoulli(temperature_relax=opt.temperature_relax)  # mask distribution
        self.p_z_notemp = IndependentRelaxedBernoulli(temperature_relax=0.001)  # mask distribution

        if self.isTrain:  # define classifiers
            print("Setting up classifier")
            self.netf_theta = init_network(
                checkpoint_path=opt.f_theta_checkpoint,
                input_shape=opt.f_theta_input_shape,
                net_module=opt.f_theta_net,
                input_nc=opt.f_theta_input_nc,
                output_classes=opt.f_theta_output_classes,
                downsample_factors=[(2, 2), (2, 2), (2, 2), (2, 2)]
                ).to(self.device)
            

        if self.isTrain:

            assert opt.lambda_regularization > 0.0, "lambda_regularization must be > 0.0"
            self.lambda_regularization = opt.lambda_regularization

            self.scheduler_lambda_regularization = get_scheduler_lambda(
                                                        scheduler_lambda_type=opt.lambda_regularization_scheduler,
                                                        target_lambda = opt.lambda_regularization,
                                                        init_lambda = opt.lambda_regularization_init,
                                                        target_epoch = opt.lambda_regularization_scheduler_targetepoch,
                                                        )
        
            self.scheduler_ising_regularization = get_scheduler_lambda(
                                                        scheduler_lambda_type = opt.lambda_ising_regularization_scheduler,  
                                                        target_lambda = opt.lambda_ising_regularization,
                                                        init_lambda = opt.lambda_ising_regularization_init,
                                                        target_epoch=opt.lambda_ising_regularization_scheduler_targetepoch,
                                                        )
     
            assert self.p_z.rsample_available, "rsample must be available for the mask distribution in order to use pathwise gradient estimator"
            
            

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_selector = torch.optim.Adam(self.netg_gamma.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_selector)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Notably, prepare the expanded images and labels used for the samples of mask distribution.
        #TODO: Could add a function when multiple pairing are available ?

        Parameters:
            input: a dictionary that contains the data itself.
        """

        self.define_nb_sample()

        self.x = input['x'].to(self.device)
        self.y = input['y'].to(self.device)

        self.x_cf = input['x_cf'].to(self.device) # TODO: @hhjs Multiple cf here ?
        self.y_cf = input['y_cf'].to(self.device) 

        # Expand the input images to match the shape of the mask samples
        self.x_cf_expanded = self.x_cf.unsqueeze(0).expand(self.sample_z, *self.x.shape)
        self.x_expanded = self.x.unsqueeze(0).expand(self.sample_z, *self.x.shape)

        self.y_expanded = self.y.unsqueeze(0).expand(self.sample_z, *self.y.shape)
        self.y_cf_expanded = self.y_cf.unsqueeze(0).expand(self.sample_z, *self.y_cf.shape)



    def define_nb_sample(self,):
        """
        Define the number of samples to draw from the mask distribution.
        Depending on the training/testing phase, the number of samples is different.
        Note that the evaluation of the likelihood is dependent on the number of mask samples.

        MC Sample: Number of samples to draw from the mask distribution that will reduce the Monte Carlo estimate variance
        IMP sample: Importance sampling to reduce the bias of the likelihood estimate (see Burda et al. 2015 Importance Weighted Autoencoders)

        """
        if self.netg_gamma.training:
            self.mc_sample_z = self.opt.mc_sample_z
            self.imp_sample_z = self.opt.imp_sample_z
        else:
            self.mc_sample_z = self.opt.mc_sample_z_notemp
            self.imp_sample_z = self.opt.imp_sample_z_notemp
        
        self.sample_z = self.mc_sample_z * self.imp_sample_z
    



    def forward(self, ):
        """Run forward pass for training; called by function <optimize_parameters>."""
        self.pi_logit = self.netg_gamma(self.x)
        if self.pi_gaussian_smoothing_sigma>0 :
            self.pi_logit = gaussian_filter_2d(self.pi_logit, sigma=self.pi_gaussian_smoothing_sigma)


        if self.upscale and not self.upscale_after_sampling :
            self.pi_logit = self.upscaler(self.pi_logit).reshape(self.x.shape[0], 1, *self.x.shape[2:])
        else :
            self.pi_logit = self.pi_logit.reshape(self.x.shape[0], 1, *[s//2**self.opt.downscale_asymmetric for s in self.x.shape[2:]])

        self.log_pi = F.logsigmoid(self.pi_logit)

        # Sample from the mask distribution
        if self.use_pi_as_mask :
            self.z = self.log_pi.exp()
            if self.upscale and self.upscale_after_sampling :
                self.z = self.upscaler(self.z)
            self.z = self.z.unsqueeze(0)
        else :
            self.z = self.p_z.rsample(self.sample_z, self.log_pi) # Need Rsample here to allow pathwise estimation
            if self.upscale and self.upscale_after_sampling :
                self.z = self.upscaler(self.z.flatten(0,1))
            self.z = self.z.reshape(self.sample_z, self.x.shape[0], 1, *self.x.shape[2:])

        if self.z_gaussian_smoothing_sigma>0 :
            self.z = gaussian_filter_2d(self.z, sigma=self.z_gaussian_smoothing_sigma)
        
        self.z = self.z.reshape(self.sample_z, self.x.shape[0], 1, *self.x.shape[2:]) 
        self.x_tilde = (self.x_expanded * self.z + (1 - self.z) * self.x_cf_expanded).flatten(0,1)
        self.y_tilde = self.netf_theta(self.x_tilde)


    def forward_val(self,):
        """Run forward pass to create all variables required for metrics and visualization; called by both functions <eval> and <test>.
        """
        # Calculate the mask distribution parameter
        self.pi_logit = self.netg_gamma(self.x)
        if self.pi_gaussian_smoothing_sigma>0 :
            self.pi_logit = gaussian_filter_2d(self.pi_logit, sigma=self.gaussian_smoothing_sigma)
        if self.upscale and not self.upscale_after_sampling :
            self.pi_logit = self.upscaler(self.pi_logit).reshape(self.x.shape[0], 1, *self.x.shape[2:])
        else :
            self.pi_logit = self.pi_logit.reshape(self.x.shape[0], 1, *[s//2**self.opt.downscale_asymmetric for s in self.x.shape[2:]])

        self.log_pi = F.logsigmoid(self.pi_logit)
        

        # Sample from the mask distribution
        self.z = self.p_z.rsample(self.sample_z, self.log_pi) # Need Rsample here to allow pathwise estimation
        self.z = self.z.reshape(self.sample_z, self.pi_logit.shape[0], 1, *self.pi_logit.shape[2:]) 


        # Sample
        self.z_notemp = self.p_z_notemp.sample(self.sample_z, self.log_pi) 
        self.z_notemp = self.z_notemp.reshape(self.sample_z, self.pi_logit.shape[0], 1, *self.pi_logit.shape[2:])
        

        # Upscale if necessary :
        if self.upscale and self.upscale_after_sampling :
            self.log_pi = self.upscaler(self.log_pi)
            self.z = self.upscaler(self.z.flatten(0,1)).reshape(self.sample_z, self.x.shape[0], 1, *self.x.shape[2:])
            self.z_notemp = self.upscaler(self.z_notemp.flatten(0,1)).reshape(self.sample_z, self.x.shape[0], 1, *self.x.shape[2:])
           
        if self.z_gaussian_smoothing_sigma>0 :
            self.z = gaussian_filter_2d(self.z, sigma=self.z_gaussian_smoothing_sigma)
            self.z_notemp = gaussian_filter_2d(self.z_notemp, sigma=self.z_gaussian_smoothing_sigma)
            self.log_pi = gaussian_filter_2d(self.log_pi, sigma=self.pi_gaussian_smoothing_sigma)
        
        # Create mixed images
        self.x_tilde_pi = (self.x_expanded * self.log_pi.exp().unsqueeze(0) + (1 - self.log_pi.exp().unsqueeze(0)) * self.x_cf_expanded).flatten(0,1)
        self.x_tilde_z = (self.x_expanded * self.z + (1 - self.z) * self.x_cf_expanded).flatten(0,1)
        self.x_tilde_notemp = (self.x_expanded * self.z_notemp + (1 - self.z_notemp) * self.x_cf_expanded).flatten(0,1)


        self.z_to_save = (self.z.flatten(0,1) * 2) -1
        self.pi_to_save = (self.log_pi.exp() * 2) -1
        self.z_to_save_notemp = (self.z_notemp.flatten(0,1) * 2) -1

        
        self.x_tilde_z = self.x_tilde_z.reshape(self.sample_z*self.x.shape[0], *self.x.shape[1:])
        self.x_tilde_notemp = self.x_tilde_notemp.reshape(self.sample_z*self.x.shape[0], *self.x.shape[1:])
        self.x_tilde_pi = self.x_tilde_pi.reshape(self.sample_z*self.x.shape[0], *self.x.shape[1:])
        
        # Calculate the classifier output on the mixed images and the original images
        self.y_tilde_z = self.netf_theta(self.x_tilde_z)
        self.y_tilde_notemp = self.netf_theta(self.x_tilde_notemp)
        self.y_no_selector = self.netf_theta(self.x)
        self.y_tilde_pi = self.netf_theta(self.x_tilde_pi)

        self.y_tilde_z = self.y_tilde_z.reshape(self.sample_z, self.x.shape[0], self.opt.f_theta_output_classes)
        self.y_tilde_notemp = self.y_tilde_notemp.reshape(self.sample_z, self.x.shape[0], self.opt.f_theta_output_classes)
        self.y_no_selector = self.y_no_selector.reshape(self.x.shape[0], self.opt.f_theta_output_classes)
        self.y_tilde_pi = self.y_tilde_pi.reshape(self.sample_z, self.x.shape[0], self.opt.f_theta_output_classes)


    def calculate_batched_loss(self):
        """
        Calculate losses, gradients, and update network weights; called in every training iteration
        """

        # Likelihood guidance 
        self.loss_class = F.cross_entropy(
            self.y_tilde.reshape(self.sample_z*self.x.shape[0], self.opt.f_theta_output_classes),
            self.y_expanded.reshape(self.sample_z*self.x.shape[0]),
            reduction='none')
        self.loss_class = self.loss_class.reshape(self.imp_sample_z, self.mc_sample_z, self.x.shape[0]).logsumexp(0).mean(0)
        
        # Regularization
        self.loss_reg =  self.z.reshape(self.sample_z, self.x.shape[0], -1).mean(0).mean(1)

        
        # Ising regularization :
        if self.lambda_ising_regularization > 0.0:
            current_z = self.z.reshape(self.sample_z, self.x.shape[0], 1, *self.x.shape[2:])
            self.loss_ising_regularization =  (current_z[:,:,:,1:] - current_z[:,:,:,:-1]).abs().flatten(2).mean(-1).mean(0) \
                                    + (current_z[:,:,:,:,1:] - current_z[:,:,:,:,:-1]).abs().flatten(2).mean(-1).mean(0) # This can be implemented with a convolution kernel ?
        else :
            self.loss_ising_regularization = torch.zeros_like(self.loss_reg)


    
    def calculate_batched_loss_val(self,):
        """
        Calculate losses and metrics using the variables created in forward val. 
        Should create all measures named in loss_names
        """
        # Regularization
        self.loss_reg =  self.z.reshape(self.sample_z, self.x.shape[0], -1).mean(0).mean(1)

        
        # Ising regularization :
        if self.lambda_ising_regularization > 0.0:
            current_z = self.z.reshape(self.sample_z, self.x.shape[0], 1, *self.x.shape[2:])
            self.loss_ising_regularization =  (current_z[:,:,:,1:] - current_z[:,:,:,:-1]).abs().flatten(2).mean(-1).mean(0) \
                                    + (current_z[:,:,:,:,1:] - current_z[:,:,:,:,:-1]).abs().flatten(2).mean(-1).mean(0) # This can be implemented with a convolution kernel
        else :
            self.loss_ising_regularization = torch.zeros_like(self.loss_reg)
        

        # Quantile metrix
        self.loss_quantile_pi_25 = self.log_pi.exp().reshape(self.x.shape[0], -1).quantile(0.25, dim=1)
        self.loss_quantile_pi_50 = self.log_pi.exp().reshape(self.x.shape[0], -1).quantile(0.50, dim=1)
        self.loss_quantile_pi_75 = self.log_pi.exp().reshape(self.x.shape[0], -1).quantile(0.75, dim=1)

        # Accuracy metrics
        self.loss_acc_z = (self.y_tilde_z.argmax(-1) == self.y_expanded).float().reshape(self.sample_z,self.x.shape[0]).mean(0)
        self.loss_acc_notemp = (self.y_tilde_notemp.argmax(-1) == self.y_expanded).float().reshape(self.sample_z,self.x.shape[0]).mean(0)
        self.loss_acc_no_selector = (self.y_no_selector.argmax(-1) == self.y).float().reshape(self.x.shape[0])
        self.loss_acc_pi = (self.y_tilde_pi.argmax(-1) == self.y_expanded).float().reshape(self.sample_z,self.x.shape[0]).mean(0)


        # Classification guidance (This is not likelihood anymore...)
        self.loss_class_pi = F.cross_entropy(
            self.y_tilde_pi.reshape(self.sample_z*self.x.shape[0], self.opt.f_theta_output_classes),
            self.y_expanded.reshape(self.sample_z*self.x.shape[0]),
            reduction='none')
        self.loss_class_pi = self.loss_class_pi.reshape(self.sample_z, self.x.shape[0]).mean(0)

        # Likelihood guidance 
        self.loss_class_z = F.cross_entropy(
            self.y_tilde_z.reshape(self.sample_z*self.x.shape[0], self.opt.f_theta_output_classes),
            self.y_expanded.reshape(self.sample_z*self.x.shape[0]),
            reduction='none')
        self.loss_class_z = self.loss_class_z.reshape(self.imp_sample_z, self.mc_sample_z, self.x.shape[0]).logsumexp(0).mean(0)
        
        
        # Likelihood guidance but without temperature relaxation
        self.loss_class_notemp = F.cross_entropy(
            self.y_tilde_notemp.reshape(self.sample_z*self.x.shape[0],self.opt.f_theta_output_classes),
            self.y_expanded.reshape(self.sample_z*self.x.shape[0]),
            reduction='none')
        self.loss_class_notemp = self.loss_class_notemp.reshape(self.imp_sample_z, self.mc_sample_z, self.x.shape[0]).logsumexp(0).mean(0)

        # Likelihood no selector
        self.loss_class_no_selector = F.cross_entropy(
            self.y_no_selector,
            self.y,
            reduction='none').reshape(self.x.shape[0])
           


    def backward_g_gamma(self):
        """Backward through the loss for the selector g_gamma"""
        # Total loss
        self.loss_total = self.loss_class.mean() 
        self.loss_total = self.loss_total + self.scheduler_lambda_regularization() * self.loss_reg.mean() \
                        + self.scheduler_ising_regularization() * self.loss_ising_regularization.mean()
        self.loss_total.backward()


    def evaluate(self):
        """
        Evaluate the model on the current batch, store and aggregate the losses.
        """
        with torch.no_grad():
            self.forward_val()
            self.calculate_batched_loss_val()
            batched_losses = self.get_current_batched_losses()
            self.aggregate_losses(batched_losses,)
       
    def update_learning_rate(self):
        """
        Modify the original learning rate function to allow for the lambda schedulers to be updated.
        """
        super().update_learning_rate()
        self.scheduler_lambda_regularization.step()
        print("Current lambda regularization : ", self.scheduler_lambda_regularization())
        self.scheduler_ising_regularization.step()
        print("Current lambda ising regularization : ", self.scheduler_ising_regularization())
        
    def get_aux_info(self):
        """
        Return a dictionary containing all the auxiliary information to be logged.
        Here returns the current regularization strenghts and the learning rate.
        """
        dic = super().get_aux_info()
        dic["lambda_regularization"] = self.scheduler_lambda_regularization()
        dic["lambda_ising_regularization"] = self.scheduler_ising_regularization()
        dic["lr"] = self.optimizer_selector.param_groups[0]['lr']
        return dic


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # Compute mask and mixed images
        self.calculate_batched_loss()
        self.set_requires_grad([self.netg_gamma,], True)
        self.optimizer_selector.zero_grad()  # set g_gamma's gradients to zero
        self.backward_g_gamma()             # calculate gradients g_gamma
        self.optimizer_selector.step()       # update g_gamma's weights
       
