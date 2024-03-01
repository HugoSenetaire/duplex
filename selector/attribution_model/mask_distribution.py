import torch.nn as nn
import torch
from abc import ABC, abstractmethod
from util.gaussian_smoothing import gaussian_filter_2d



def get_mask_distribution(mask_distribution_name):
    """
    Create a mask distribution.

    Parameters:
        opt (argparse.Namespace): options

    Returns:
        AbstractMaskDistribution: mask distribution
    """
    if mask_distribution_name == "id_rbernoulli":
        current_mask_distribution = IndependentRelaxedBernoulli
    else:
        raise ValueError("Mask distribution {} not recognized".format(mask_distribution_name))

    return current_mask_distribution

class AbstractMaskDistribution(ABC, nn.Module):
    """
    Abstract base class for mask distributions.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train, opt=None):
        """Add new trainer-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions. Used here for commandline option in attributes of trainer
        Returns:
            the modified parser.
        """
        return parser

    def __init__(self, opt,):
        super(AbstractMaskDistribution, self).__init__()
        self.current_distribution = None
        self.rsample_available = False
        self.allow_pi_as_mask = False

    def set_parameter(self, g_gamma_out):
        """
        Set the parameter of the mask distribution.

        Parameters:
            g_gamma_out (torch.Tensor): output of the generator. Shape (batch_size, *dim) # Maybe I should use a dictionnary here for future updates
 
        Returns:
            None
        """
        pass
        
    def get_attribution_score(self, g_gamma_out):
        """
        Get the attribution score of the mask distribution. This is defined as the expectation
        of the mask distribution.

        Parameters:
            g_gamma_out (torch.Tensor): output of the generator. Shape (batch_size, *dim)

        Returns:
            torch.Tensor: attribution score of the mask distribution. Shape (batch_size, *dim)
        """
        pass

    def sample(self, nb_sample, g_gamma_out = None):
        """
        Sample from the mask distribution.
    
        Parameters:
            nb_sample (int): number of samples to draw from the distribution.

        Returns:
            torch.Tensor: samples from the mask distributions
        """
        pass

    def rsample(self, nb_sample, g_gamma_out = None):
        """
        Sample from the mask distribution.
    
        Parameters:
            nb_sample (int): number of samples to draw from the distribution.

        Returns:
            torch.Tensor: samples from the mask distributions
        """
        raise ValueError("rsample not implemented for this distribution")

    def log_prob(self, z, g_gamma_out = None):
        """
        Compute the log probability of the mask distribution.
    
        Parameters:
            z (torch.Tensor): samples from the mask distribution. Shape (batch_size, nb_samples, *dim)

        Returns:
            torch.Tensor: log probability of the samples from the mask distribution. Shape (batch_size, nb_samples)
        """
        pass



class IndependentRelaxedBernoulli(AbstractMaskDistribution):
    """
    Independent Relaxed Bernoulli distribution.
    Name of the distribution: id_rbernoulli
    """
    @staticmethod
    def modify_commandline_options(parser, is_train, opt=None):
        """Add new trainer-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--temperature_relax', type=float, default=1.0, help='temperature for the relaxed bernoulli distribution')
        parser.add_argument('--z_gaussian_smoothing_sigma', type=float, default=-1.0, help='if specified,\
                            the mask distribution output will be smoothed using a gaussian filter of this standard deviation')
        return parser

    def __init__(self, opt) -> None:
        super().__init__(opt,)

        self.distribution = torch.distributions.relaxed_bernoulli.RelaxedBernoulli
        self.temperature_relax = opt.temperature_relax
        self.current_distribution = None
        self.rsample_available = True
        self.allow_pi_as_mask = True
        self.z_gaussian_smoothing_sigma = opt.z_gaussian_smoothing_sigma
       
    def set_parameter(self, g_gamma_out):
        """
        Set the parameter of the mask distribution.
        """
        self.current_distribution = self.distribution(temperature=self.temperature_relax, logits = g_gamma_out)

    def sample(self, nb_sample, g_gamma_out = None):
        if g_gamma_out is not None:
            self.set_parameter(g_gamma_out)

        sample_z = self.current_distribution.sample((nb_sample,))
        if self.z_gaussian_smoothing_sigma > 0.:
            sample_z = gaussian_filter_2d(sample_z.flatten(0,1), sigma = self.z_gaussian_smoothing_sigma)
        sample_z = sample_z.reshape(nb_sample, *g_gamma_out.shape)
        return sample_z
    
    def rsample(self, nb_sample, g_gamma_out = None):
        if g_gamma_out is not None:
            self.set_parameter(g_gamma_out)

        sample_z = self.current_distribution.rsample((nb_sample,))
        if self.z_gaussian_smoothing_sigma > 0.:
            sample_z = gaussian_filter_2d(sample_z.flatten(0,1), sigma = self.z_gaussian_smoothing_sigma)
        sample_z = sample_z.reshape(nb_sample, *g_gamma_out.shape)
        return sample_z
    
    def get_log_pi(self, g_gamma_out):
        """
        Given the mask distribution parameters, return the log probability of the mask distribution.
        In the case of the Independent Relaxed Bernoulli, this is directly the log-simoid of the parameters.
        """
        log_pi = torch.nn.functional.logsigmoid(g_gamma_out)
        return log_pi



    def log_prob(self, z, g_gamma_out=None):
        if g_gamma_out is not None:
            self.set_parameter(g_gamma_out)
        
        return self.current_distribution.log_prob(z).sum(dim=-1)