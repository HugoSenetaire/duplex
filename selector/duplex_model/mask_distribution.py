import torch.nn as nn
import torch


class AbstractMaskDistribution(nn.Module):
    """
    Abstract base class for mask distributions.
    """

    def __init__(self, ):
        super(AbstractMaskDistribution, self).__init__()
        self.current_distribution = None
        self.rsample_available = False

    def set_parameter(self, g_gamma_out):
        """
        Set the parameter of the mask distribution.

        Parameters:
            g_gamma_out (torch.Tensor): output of the generator. Shape (batch_size, *dim) # Maybe I should use a dictionnary here for future updates
 
        Returns:
            None
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
    """

    def __init__(self, temperature_relax = 0.1 ) -> None:
        super().__init__()
        self.distribution = torch.distributions.relaxed_bernoulli.RelaxedBernoulli
        self.temperature_relax = temperature_relax
        self.current_distribution = None
        self.rsample_available = True
       
    def set_parameter(self, g_gamma_out):
        """
        Set the parameter of the mask distribution.
        """
        self.current_distribution = self.distribution(temperature=self.temperature_relax, logits = g_gamma_out)

    def sample(self, nb_sample, g_gamma_out = None):
        if g_gamma_out is not None:
            self.set_parameter(g_gamma_out)
        
        return self.current_distribution.sample((nb_sample,))
    
    def rsample(self, nb_sample, g_gamma_out = None):
        if g_gamma_out is not None:
            self.set_parameter(g_gamma_out)
        
        return self.current_distribution.rsample((nb_sample,))



    def log_prob(self, z, g_gamma_out=None):
        if g_gamma_out is not None:
            self.set_parameter(g_gamma_out)
        
        return self.current_distribution.log_prob(z).sum(dim=-1)