import torch
import torch.nn.functional as F
from duplex_trainer.base_trainer import BaseTrainer
from attribution_model import initAttributionModel
from duplex_trainer.scheduler_lambda import get_scheduler_lambda


class PathWiseTrainer(BaseTrainer):
    """
    This class implements the pathwise selector trainer for 
    a DupLEX models. It requires a selector network and a classifier network and 
    a dataset that provides counterfactuals. 
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
        parser.add_argument('--trainer_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[trainer_suffix].pth will be loaded as the generator.')
        parser.add_argument('--use_pi_as_mask', action='store_true', help='If specified, the mask distribution is not sampled and we directly optimize on pi. \
                                                                        This is only possible when the imputation method is deterministic (or very simple).')

        parser.add_argument('--use_classifier_target', action='store_true', help='If specified, the selector is trained to predict the classifier output on the original image instead of the dataset target.')

        parser.add_argument('--lambda_regularization', type=float, default=1.0, help='L1 regularization strenght to limit the selection of the mask')
        parser.add_argument('--lambda_regularization_init', type=float, default=0., help='Initial value for the lambda_regularization scheduler')
        parser.add_argument('--lambda_regularization_scheduler', type=str, default='constant', help='Scheduler for the lambda_regularization parameter. [linear | constant | cosine]')
        parser.add_argument('--lambda_regularization_scheduler_targetepoch', type=int, default=10, help='Target epoch for the lambda_regularization scheduler to reach the lambda_regularization value')

        parser.add_argument('--lambda_ising_regularization', type=float, default=-1.0, help='Ising regularization strenght to enforce connectivity in the mask selection')
        parser.add_argument('--lambda_ising_regularization_init', type=float, default=0., help='Initial value for the lambda_ising_regularization scheduler')
        parser.add_argument('--lambda_ising_regularization_scheduler', type=str, default='constant', help='Scheduler for the lambda_ising_regularization parameter. [linear | constant | cosine]')
        parser.add_argument('--lambda_ising_regularization_scheduler_targetepoch', type=int, default=10, help='Target epoch for the lambda_ising_regularization scheduler to reach the lambda_ising_regularization value')
        
        return parser

    def __init__(self, opt):
        """Initialize the Selector class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseTrainer.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <Basetrainer.get_current_losses>
        self.loss_names = ["ising_regularization", "reg", "class_z", "class_no_selector", "class_pi", "acc_z", "acc_no_selector", "acc_pi", "quantile_pi_25", "quantile_pi_50", "quantile_pi_75"]

        # specify the images you want to save/display. The training/test scripts will call <Basetrainer.get_current_visuals>
        self.visual_names = ['x_expanded', 'x_cf_expanded', 'pi_to_save', 'x_tilde_pi', 'z_to_save', 'x_tilde_z', ]

        self.use_pi_as_mask = opt.use_pi_as_mask
        self.use_classifier_target = opt.use_classifier_target
        self.z_gaussian_smoothing_sigma = opt.z_gaussian_smoothing_sigma

        self.lambda_ising_regularization = opt.lambda_ising_regularization



        


        # define networks (both selectors and classifiers)
        print("Setting Duplex")
        self.duplex = initAttributionModel(opt)
        self.trainer_names = ['selector']
        self.classifier = self.duplex.classifier
        self.selector = self.duplex.selector
        self.p_z = self.duplex.mask_distribution


        if self.use_pi_as_mask :
            assert self.duplex.mask_distribution.allow_pi_as_mask, "The mask distribution does not allow pi as mask. Please set use_pi_as_mask to False"


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
            
            

        # initialize optimizers; schedulers will be automatically created by function <Basetrainer.setup>.
            self.optimizer_selector = torch.optim.Adam(self.selector.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_selector)

    def set_target(self,):
        """Set the target for the selector. 
        The target is the classifier output on the original image if use_classifier_target is specified, 
        otherwise it is the dataset target.
        """
        if self.use_classifier_target:
            with torch.no_grad():
                self.y_expanded = torch.distributions.Categorical(probs=self.classifier(self.x_expanded.flatten(0,1)).softmax(-1)).sample()
                self.y_expanded = self.y_expanded.reshape(self.sample_z, self.x.shape[0])
                self.y = self.y_expanded[0]
        else :
            self.y = self.y_expanded[0]
            self.y_expanded = self.y.unsqueeze(0).expand(self.sample_z, *self.y.shape)

        
       

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Notably, prepare the expanded images and labels used for the samples of mask distribution.
        #TODO: Could add a function when multiple pairing are available ?

        Parameters:
            input: a dictionary that contains the data itself.
        """

        self.define_nb_sample()

        self.x_path = input['x_path']
        self.x_cf_path = input['x_cf_path']

        self.x = input['x'].to(self.device)      
        self.x_cf = input['x_cf'].to(self.device) # TODO: @hhjs Multiple cf here ?
        self.y_cf = input['y_cf'].to(self.device) 
        self.y_cf_expanded = self.y_cf.unsqueeze(0).expand(self.sample_z, *self.y_cf.shape)

        # Expand the input images to match the shape of the mask samples
        self.x_cf_expanded = self.x_cf.unsqueeze(0).expand(self.sample_z, *self.x.shape)
        self.x_expanded = self.x.unsqueeze(0).expand(self.sample_z, *self.x.shape)

        self.set_target()

        self.real_y_cf = self.classifier(self.x_cf).softmax(-1).reshape(self.x_cf.shape[0], self.opt.f_theta_output_classes)



    def define_nb_sample(self,):
        """
        Define the number of samples to draw from the mask distribution.
        Depending on the training/testing phase, the number of samples is different.
        Note that the evaluation of the likelihood is dependent on the number of mask samples.

        MC Sample: Number of samples to draw from the mask distribution that will reduce the Monte Carlo estimate variance
        IMP sample: Importance sampling to reduce the bias of the likelihood estimate (see Burda et al. 2015 Importance Weighted Autoencoders)

        """
        if not self.eval_mode:
            self.mc_sample_z = self.opt.mc_sample_z
            self.imp_sample_z = self.opt.imp_sample_z
        else:
            self.mc_sample_z = self.opt.mc_sample_z_test
            self.imp_sample_z = self.opt.imp_sample_z_test
        
        self.sample_z = self.mc_sample_z * self.imp_sample_z
    



    def forward(self, ):
        """Run forward pass for training; called by function <optimize_parameters>."""
        self.mask_distribution_input = self.duplex._get_mask_param(
                                                self.x_expanded.flatten(0,1),
                                                self.x_cf_expanded.flatten(0,1),
                                                self.y_expanded.flatten(0,1),
                                                self.y_cf_expanded.flatten(0,1),
                                                )


        self.log_pi_expanded = self.duplex.mask_distribution.get_log_pi(self.mask_distribution_input)

        # Sample from the mask distribution
        if self.use_pi_as_mask :
            self.z_expanded = self.log_pi_expanded.exp()
        else :
            self.z_expanded = self.p_z.rsample(1, self.log_pi_expanded).reshape(self.log_pi_expanded.shape) # Need Rsample here to allow pathwise estimation
        
        self.z_expanded = self.z_expanded.reshape(self.sample_z, self.x.shape[0], 1, *self.x.shape[2:]) 
        self.x_tilde_expanded = (self.x_expanded * self.z_expanded + (1 - self.z_expanded) * self.x_cf_expanded).flatten(0,1)
        self.y_tilde_expanded = self.classifier(self.x_tilde_expanded)


    def forward_val(self,):
        """Run forward pass to create all variables required for metrics and visualization; called by both functions <eval> and <test>.
        """
        # Calculate the mask distribution parameter
        
        self.mask_distribution_input = self.duplex._get_mask_param(
                                                self.x_expanded.flatten(0,1),
                                                self.x_cf_expanded.flatten(0,1),
                                                self.y_expanded.flatten(0,1),
                                                self.y_cf_expanded.flatten(0,1),
                                            )
        self.log_pi_expanded = self.duplex.mask_distribution.get_log_pi(self.mask_distribution_input)

        

        # Sample from the mask distribution with temperature relaxation
        self.z_expanded = self.p_z.rsample(1, self.log_pi_expanded) # Need Rsample here to allow pathwise estimation
        self.z_expanded = self.z_expanded.reshape(self.sample_z, self.x.shape[0], 1, *self.x.shape[2:])
        self.log_pi_expanded = self.log_pi_expanded.reshape(self.sample_z, self.x.shape[0], 1, *self.x.shape[2:])
        
        # Create mixed images
        self.x_tilde_pi = (self.x_expanded * self.log_pi_expanded.exp() + (1 - self.log_pi_expanded.exp()) * self.x_cf_expanded)
        self.x_tilde_z = (self.x_expanded * self.z_expanded + (1 - self.z_expanded) * self.x_cf_expanded)

        # Put the mask in the right format for visualization
        self.z_to_save = (self.z_expanded * 2) -1
        self.pi_to_save = (self.log_pi_expanded.exp() * 2) -1

 
        
        # Calculate the classifier output on the mixed images and the original images
        self.y_tilde_z = self.classifier(self.x_tilde_z.reshape(self.sample_z*self.x.shape[0], *self.x.shape[1:]))
        self.y_no_selector = self.classifier(self.x.reshape(self.sample_z*self.x.shape[0], *self.x.shape[1:]))
        self.y_tilde_pi = self.classifier(self.x_tilde_pi.reshape(self.sample_z*self.x.shape[0], *self.x.shape[1:]))

        self.y_tilde_z = self.y_tilde_z.reshape(self.sample_z, self.x.shape[0], self.opt.f_theta_output_classes)
        self.y_no_selector = self.y_no_selector.reshape(self.x.shape[0], self.opt.f_theta_output_classes)
        self.y_tilde_pi = self.y_tilde_pi.reshape(self.sample_z, self.x.shape[0], self.opt.f_theta_output_classes)




    def calculate_batched_loss(self):
        """
        Calculate losses, gradients, and update network weights; called in every training iteration
        """

        # Likelihood guidance 
        self.loss_class = F.cross_entropy(
            self.y_tilde_expanded.reshape(self.sample_z*self.x.shape[0], self.opt.f_theta_output_classes),
            self.y_expanded.reshape(self.sample_z*self.x.shape[0]),
            reduction='none')
        self.loss_class = self.loss_class.reshape(self.imp_sample_z, self.mc_sample_z, self.x.shape[0]).logsumexp(0).mean(0)
        
        # Regularization
        self.loss_reg =  self.z_expanded.reshape(self.sample_z, self.x.shape[0], -1).mean(0).mean(1)

        
        # Ising regularization :
        if self.lambda_ising_regularization > 0.0:
            current_z = self.z_expanded.reshape(self.sample_z, self.x.shape[0], 1, *self.x.shape[2:])
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
        self.loss_reg =  self.z_expanded.reshape(self.sample_z, self.x.shape[0], -1).mean(0).mean(1)

        
        # Ising regularization :
        if self.lambda_ising_regularization > 0.0:
            current_z = self.z_expanded.reshape(self.sample_z, self.x.shape[0], 1, *self.x.shape[2:])
            self.loss_ising_regularization =  (current_z[:,:,:,1:] - current_z[:,:,:,:-1]).abs().flatten(2).mean(-1).mean(0) \
                                    + (current_z[:,:,:,:,1:] - current_z[:,:,:,:,:-1]).abs().flatten(2).mean(-1).mean(0) # This can be implemented with a convolution kernel
        else :
            self.loss_ising_regularization = torch.zeros_like(self.loss_reg)
        

        # Quantile metrix
        self.loss_quantile_pi_25 = self.log_pi_expanded.exp().reshape(self.x.shape[0], -1).quantile(0.25, dim=1)
        self.loss_quantile_pi_50 = self.log_pi_expanded.exp().reshape(self.x.shape[0], -1).quantile(0.50, dim=1)
        self.loss_quantile_pi_75 = self.log_pi_expanded.exp().reshape(self.x.shape[0], -1).quantile(0.75, dim=1)

        # Accuracy metrics
        self.loss_acc_z = (self.y_tilde_z.argmax(-1) == self.y_expanded).float().reshape(self.sample_z,self.x.shape[0]).mean(0)
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

        # Likelihood no selector
        self.loss_class_no_selector = F.cross_entropy(
            self.y_no_selector,
            self.y,
            reduction='none').reshape(self.x.shape[0])
           


    def backward_g_gamma(self):
        """Backward through the loss for the selector g_gamma"""
        # Total loss
        self.loss_total = self.loss_class.mean() 
        self.loss_total = self.loss_total \
                        + self.scheduler_lambda_regularization() * self.loss_reg.mean() \
                        + self.scheduler_ising_regularization() * self.loss_ising_regularization.mean()
        self.loss_total.backward()


    def evaluate(self):
        """
        Evaluate the trainer on the current batch, store and aggregate the losses.
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
        self.set_requires_grad([self.selector,], True)
        self.optimizer_selector.zero_grad()  # set g_gamma's gradients to zero
        self.backward_g_gamma()             # calculate gradients g_gamma
        self.optimizer_selector.step()       # update g_gamma's weights
       

    def set_input_fix(self, input, target_cf):
        raise NotImplementedError("set_input_fix does not make sense for such trainer. Use set_input instead.")
