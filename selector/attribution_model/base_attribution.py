from abc import ABC, abstractmethod

class BaseAttribution(ABC):
    # TODO use ABC?
    """
    Basic format of an attribution class.
    """
    def __init__(self, classifier):
        self.classifier = classifier
    
    def _attribute(self, real_img, counterfactual_img, real_class, target_class, **kwargs):
        raise NotImplementedError("The base attribution class does not have an attribute method.")

    def attribute(self, real_img, counterfactual_img, real_class, target_class, **kwargs):
        self.classifier.zero_grad()
        attribution = self._attribute(real_img, counterfactual_img, real_class, target_class, **kwargs)
        return attribution.detach().cpu().numpy()