"""
A dataset for synapse data loader from the generated gan pair dataset.
As oppose to synapsepairfolder_dataset, this dataset returns the original 
and **all** counterfactual images in a dictionary.
"""
import logging
from typing import Any
import numpy as np
from torch.utils.data import IterableDataset
from data.base_dataset import BaseDataset
from torch.utils.data import Dataset
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from .utils_pair_image_folder import make_dataset, find_classes, is_image_file, default_loader
import albumentations as A
import albumentations.pytorch as apytorch
import os


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class SynapseGanCFDataset(BaseDataset,):
    """
    A dataset for which we return the original and **all** counterfactual images.
    Return a dummy counterfactual target since all of them are stored in the same tensor.
    Then the selection will happen in a specific pathwise selector model.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        BaseDataset.modify_commandline_options(parser, is_train)
        parser.add_argument("--dataroot_counterfactual", type=str, default="/nrs/funke/adjavond/data/duplex/cyclegan/", help="path to the counterfactual dataset")
        return parser

    def __init__(
        self,
        opt,
        split="train",
        ):
        """
        Parameters:
            root: root directory of the dataset
        """
        super().__init__(opt)

        source_directory = os.path.join(opt.dataroot, split)
        paired_directory = os.path.join(opt.dataroot_counterfactual, split)
        classes, class_to_idx = find_classes(source_directory)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples, self.samples_dic = make_dataset(source_directory, paired_directory, class_to_idx, is_valid_file=is_image_file,)
        # self.transform = transform
        if self.augment and self.split == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    apytorch.transforms.ToTensorV2(),
                ],
                additional_targets={str(key): "image" for key in self.class_to_idx.values()},
            )
        else:
            self.transform = A.Compose(
            [
                apytorch.transforms.ToTensorV2()
            ],
            additional_targets={str(key): "image" for key in self.class_to_idx.values()},
        )


    def __len__(self):
        return len(self.samples_dic)

    def __getitem__(self, index):
        paths, target = self.samples_dic[index]
        current_sample = default_loader(paths[str(target)])
        counterfactuals = {str(key): default_loader(paths[str(key)]) for key in self.class_to_idx.values()}
        transformed = self.transform(image=current_sample, **counterfactuals)

        total_cf = []
        for key in self.class_to_idx.values():
            total_cf.append(transformed[str(key)])
        # transformed["total_cf"] = torch.stack(total_cf)
        dic = {
            "x": transformed["image"],
            "y": torch.tensor(target),
            "x_cf": torch.stack(total_cf),
            "y_cf": torch.tensor(-1),
        }
        return dic
        