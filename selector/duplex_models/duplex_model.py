from .base_attribution import BaseAttribution

class DupLEX(BaseAttribution):
    """
    DupLEX attribution method.
    """
    def __init__(self, classifier, selector,):
        super(DupLEX, self).__init__(classifier)
        self.selector = selector

    def _attribute(self, real_img, counterfactual_img, real_class, target_class, **kwargs):
        """
        Compute the DupLEX attribution.
        """
        attribution = self.selector(real_img, counterfactual_img, real_class, target_class,)
        return attribution
    