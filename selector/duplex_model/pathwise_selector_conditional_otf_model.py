from duplex_model.pathwise_selector_otf_model import PathWiseSelectorOTFModel
from util.gaussian_smoothing import gaussian_filter_2d
import torch
from torchinfo import summary
import torch.nn.functional as F


class PathwiseSelectorConditionalOTFModel(PathWiseSelectorOTFModel):
    """
    Class that conditions the pathwise selector model on the counterfactual image.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        PathWiseSelectorOTFModel.modify_commandline_options(parser, is_train)
        # TODO any new options?
        return parser

    def __init__(self, opt):
        # Double the input channels
        setattr(opt, 'input_nc', 2 * opt.input_nc)
        super().__init__(opt)

    def set_input(self, input):
        super().set_input(input)
        self.netg_input =  torch.cat([self.x, self.x_cf], dim=1) # Concatenate on the channel dimension

    def forward(self, ):
        """Run forward pass for training; called by function <optimize_parameters>."""
        # TODO @adjavon copied the forward function from the parent because self.x is used may places after this. Could simplify
        self.pi_logit = self.netg_gamma(self.netg_input)
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
        self.pi_logit = self.netg_gamma(self.netg_input)
        if self.pi_gaussian_smoothing_sigma>0 :
            self.pi_logit = gaussian_filter_2d(self.pi_logit, sigma=self.pi_gaussian_smoothing_sigma)
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
            self.log_pi = gaussian_filter_2d(self.log_pi, sigma=self.z_gaussian_smoothing_sigma)
        
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
