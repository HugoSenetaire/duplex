from .base_attribution import BaseAttribution
from util.gaussian_smoothing import gaussian_filter_2d
import torch

    
    

class DupLEX(BaseAttribution):
    """
    DupLEX is a method that uses a selector and mask distribution to compute the attribution of a classifier.

    Args:
        classifier (nn.Module): The classifier to attribute.
        selector (nn.Module): A selector network that outputs mask distribution parameters.
        mask_distribution (MaskDistribution): A mask distribution that takes in the selector output and one can sample a mask from.
        upscaler (nn.Module): A network that upscales the mask distribution output to the input size of the classifier.
        renormalization_module (nn.Module): A module that renormalizes the classifier input to the range of the input data (for instance,\
            StarGAN might be trained for [0,1] and the classifier on Resnet)
        use_counterfactual_as_input (bool): Whether to use the counterfactual image as input to the selector.
    """
    def __init__(self,
                classifier,
                selector,
                mask_distribution,
                upscaler=None,
                renormalization_module=None,
                use_counterfactual_as_input=False,
                param_gaussian_smoothing_sigma = False,
                ):
        super(DupLEX, self).__init__(classifier)
        self.selector = selector
        self.mask_distribution = mask_distribution
        self.use_counterfactual_as_input = use_counterfactual_as_input
        self.upscaler = upscaler
        self.renormalization_module = renormalization_module
        if self.renormalization_module is not None:
            self.classifier = torch.nn.Sequential(self.renormalization_module, self.classifier)
        self.param_gaussian_smoothing_sigma = param_gaussian_smoothing_sigma
        
        


    def _get_mask_param(self, real_img, counterfactual_img, real_class, target_class,):
        """
        Get the mask distribution parameters using the selector.
        """
        if self.use_counterfactual_as_input:
            input = torch.cat([real_img, counterfactual_img], dim=1)
        else :
            input = real_img
        mask_input = self.selector(input)

        if self.upscaler is not None:
            mask_input = self.upscaler(mask_input)

        if self.param_gaussian_smoothing_sigma > 0.:
            mask_input = gaussian_filter_2d(mask_input, sigma = self.param_gaussian_smoothing_sigma,)

        return mask_input

    def _attribute(self, real_img, counterfactual_img, real_class, target_class,):
        """
        Compute the DupLEX attribution using the selector and mask distribution.
        """
        if len(real_img.shape) == 3:
            real_img = real_img.unsqueeze(0)
            counterfactual_img = counterfactual_img.unsqueeze(0)
        
        mask_param = self._get_mask_param(real_img, counterfactual_img, real_class, target_class,)
        attribution = self.mask_distribution.get_log_pi(mask_param).exp()

        return attribution
    