"""On-the-fly synapse data generator."""
import logging
from typing import Any
import numpy as np
from torch.utils.data import IterableDataset
from data.base_dataset import BaseDataset
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import albumentations as A
import albumentations.pytorch as apytorch
import os


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class CounterfactualNotFound(Exception):
    pass


class SynapseNoCFDataset(BaseDataset,):
    """
    A dataset for which a dummy counterfactual is returned along the true image.
    A target for the counterfactual is generated and returned.
    It assumes the images can be loaded with the ImageFolder dataset.
    """
    def __init__(
        self,
        opt,
        split="train",
        ):

        """
        Parameters:
            root: root directory of the dataset
        """
        super().__init__(opt, split)
        root = os.path.join(opt.dataroot, split)
        self.augment = (not opt.no_augment)
        self.split = split


        # Dataset
        self.dataset = ImageFolder(
            root=root,
            # transform=None,
            transform=transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5),
                ]
            ),
        )


    def __make_dict(self, x, xcf, y, ycf):
        return {
            "x": x,
            "x_cf": xcf,
            "y": y,
            "y_cf": ycf,
            # "x_path": None,
            # "xcf_path": None
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        self.dataset.__getitem__(index)
        x, y = self.dataset.__getitem__(index)
        y = torch.tensor(y, dtype=torch.long)
        # target = np.random.choice([i for i in range(1, 6) if i != y])
        target = torch.full_like(y, -1)
        x_cf = torch.full_like(x, -1)
        return self.__make_dict(x, x_cf, y, target)
