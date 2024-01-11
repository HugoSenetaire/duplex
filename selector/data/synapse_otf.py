"""On-the-fly synapse data generator."""
import logging
import numpy as np
from torch.utils.data import IterableDataset
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

from funlib.learn.torch.models import Vgg2D
from starganv2.inference.model import LatentInferenceModel

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class CounterfactualNotFound(Exception):
    pass


class StarganDupLEX(IterableDataset):
    """
    A dataset of counterfactuals generated on-the-fly using the Stargan-v2 model.
    It assumes the images can be loaded with the ImageFolder dataset.
    """
    def __init__(
        self,
        root: str,
        latent_model_checkpoint_dir: str,
        classifier_checkpoint: str,
        img_size: int = 128,
        style_dim: int = 64,
        latent_dim: int = 16,
        num_domains: int = 6,
        checkpoint_iter: int = 100000,
        classifier_fmaps: int = 12,
    ):
        """
        Parameters:
            root: root directory of the dataset
            latent_model_checkpoint_dir: directory with the latent model checkpoint
            classifier_checkpoint: path to the classifier checkpoint
            img_size: image size, the image is assumed to be 2D
            style_dim: style dimension of the Stargan model
            latent_dim: latent dimension of the Stargan model
            num_domains: number of classes or domains
            checkpoint_iter: checkpoint of the Stagan model to load
            classifier_fmaps: number of feature maps of the classifier
        """
        # Inference model
        self.latent_inference_model = LatentInferenceModel(
            checkpoint_dir=latent_model_checkpoint_dir,
            img_size=img_size,
            style_dim=style_dim,
            latent_dim=latent_dim,
            num_domains=num_domains,
            w_hpf=0.0,
        )
        self.latent_inference_model.load_checkpoint(checkpoint_iter)
        # Classifier
        self.classifier = Vgg2D(input_size=(img_size, img_size), fmaps=classifier_fmaps)
        self.classifier.load_state_dict(
            torch.load(classifier_checkpoint)["model_state_dict"]
        )
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_inference_model.to(self.device)
        self.classifier.to(self.device)
        # Eval
        self.latent_inference_model.eval()
        self.classifier.eval()

        # Dataset
        # TODO make transforms configurable
        self.dataset = ImageFolder(
            root=root,
            transform=transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5),
                ]
            ),
        )

    @torch.no_grad()
    def get_counterfactual(self, x, target):
        """
        Tries to find a counterfactual for the given sample, given the target.
        It creates a batch, and returns one of the samples if it is classified correctly.

        params: x: sample
        params: target: target class
        returns: counterfactual
        raises: CounterfactualNotFound
        """
        # Copy x batch_size times
        x_multiple = torch.stack([x] * self.batch_size)
        # Generate batch_size counterfactuals
        xcf = self.latent_inference_model(
            x_multiple.to(self.device),
            torch.tensor([target] * self.batch_size).to(self.device),
        )
        # Evaluate the counterfactuals
        p = torch.softmax(self.classifier(xcf), dim=-1)
        # Get the predictions
        predictions = torch.argmax(p, dim=-1)
        # Get the indices of the correct predictions
        indices = torch.where(predictions == target)[0]
        if len(indices) == 0:
            raise CounterfactualNotFound()
        # Choose one of the correct predictions
        index = np.random.choice(indices.cpu().numpy())
        # Get the counterfactual
        xcf = xcf[index].cpu().numpy()
        logger.info(f"Number of counterfactuals: {len(indices)}")
        return xcf

    def __make_dict(self, x, xcf, y, ycf):
        return {
            "x": x,
            "xcf": xcf,
            "y": y,
            "ycf": ycf,
            "x_path": None,
            "xcf_path": None
        }

    def __iter__(self):
        for idx, (x, y) in enumerate(self.dataset):
            target = np.random.choice([i for i in range(1, 6) if i != y])
            try:
                xcf = self.get_counterfactual(x, target)
                yield self.__make_dict(x, xcf, y, target)
            except CounterfactualNotFound:
                logger.info(
                    f"Counterfactual not found for sample {idx} with target {target}."
                )
                continue
