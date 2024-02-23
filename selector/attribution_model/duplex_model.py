from .base_attribution import BaseAttribution
import torch
    
    



class DupLEX(BaseAttribution):
    """
    DupLEX is a method that uses a selector and mask distribution to compute the attribution of a classifier.

    Args:
        classifier (nn.Module): The classifier to attribute.
        selector (nn.Module): A selector network that outputs mask distribution parameters.
        mask_distribution (MaskDistribution): A mask distribution that takes in the selector output and one can sample a mask from.
        use_counterfactual_as_input (bool): Whether to use the counterfactual image as input to the selector.
    """
    def __init__(self, classifier, selector, mask_distribution, use_counterfactual_as_input=False,):
        super(DupLEX, self).__init__(classifier)
        self.selector = selector
        self.mask_distribution = mask_distribution
        self.use_counterfactual_as_input = use_counterfactual_as_input

    def _get_mask_param(self, real_img, counterfactual_img, real_class, target_class,):
        """
        Get the mask distribution parameters using the selector.
        """
        if self.use_counterfactual_as_input:
            input = torch.cat([real_img, counterfactual_img], dim=1)
        else :
            input = real_img
        mask_param = self.selector(input)

        return mask_param

    def _attribute(self, real_img, counterfactual_img, real_class, target_class,):
        """
        Compute the DupLEX attribution using the selector and mask distribution.
        """
        
        mask_param = self._get_mask_param(real_img, counterfactual_img, real_class, target_class,)
        attribution = self.mask_distribution.get_attribution_score(mask_param)

        return attribution
    