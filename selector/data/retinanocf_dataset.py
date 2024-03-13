"""On-the-fly synapse data generator."""
import logging
from typing import Any
import numpy as np
from data.base_dataset import BaseDataset
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import albumentations as A
import albumentations.pytorch as apytorch
from PIL import Image


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class CounterfactualNotFound(Exception):
    pass

class ImageFolderAugmentedDataset(ImageFolder,):
    """
    Similar to Image Folder, but returns the path of the original image.
    """
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(image=sample)["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path


class RetinaNoCFDataset(BaseDataset,):
    """
    A dataset for which a dummy counterfactual and a dummy target is returned along the true image.
    It assumes the images can be loaded with the ImageFolder dataset.
    """

    def __scale(self, x, target, method=Image.BICUBIC):
        ow, oh = x.size
        if (ow == target):
            return x
        w = target
        h = target
        return x.resize((w, h), method)

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

        # Dataset

        if self.augment and self.split == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    apytorch.transforms.ToTensorV2(),

                ],

            )
        else:
            self.transform = A.Compose(
            [
                apytorch.transforms.ToTensorV2()
            ],
        )

        self.dataset = ImageFolderAugmentedDataset(
            root=root,
            transform=None,
        )

        self.idx_to_class = {k: k for k in self.dataset.classes}


    def __make_dict(self, x, xcf, y, ycf, x_path = "None",):
        return {
            "x": x,
            "x_cf": xcf,
            "y": y,
            "y_cf": ycf,
            "x_path": x_path,
            "x_cf_path": "None",
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y, path = self.dataset.__getitem__(index)
        x = self.__scale(x, self.opt.load_size)
        x = np.array(x, dtype=np.float32) # It's a grayscale image, so we only need one channel
        x = self.transform(image=x)["image"]/255 *2 -1 # Normalize to [-1, 1]
        y = torch.tensor(y, dtype=torch.long)
        # target = np.random.choice([i for i in range(1, 6) if i != y])
        target = torch.full_like(y, -1)
        x_cf = torch.full_like(x, -1)
        return self.__make_dict(x, x_cf, y, target, x_path=path)
