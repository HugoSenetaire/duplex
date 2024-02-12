"""DEPRECATED: A dataset for synapse data loader from the prototype dataset.

The data is expected to be organized as `{root}/{image_type}/{class_name}/{sample_number}.png` where: 
- `root_dir`
- `image_type` is either `original` or `counterfactual`
- `class_name` is the name of the original class
- `sample_number` is the number of the sample.
"""
import albumentations as A
import albumentations.pytorch as apytorch
import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from data.base_dataset import BaseDataset
import torch

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(
    dir,
    counterfactual_dir,
    max_dataset_size=float("inf"),
):
    images = []
    counterfactuals = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir
    assert os.path.isdir(counterfactual_dir), (
        "%s is not a valid directory" % counterfactual_dir
    )

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                # Check the counterfactual directory for the same file
                counterfactual_path = path.replace("original", "counterfactual")
                if is_image_file(counterfactual_path):
                    images.append(path)
                    counterfactuals.append(counterfactual_path)
    return (
        images[: min(max_dataset_size, len(images))],
        counterfactuals[: min(max_dataset_size, len(images))],
    )

from .utils_pair_image_folder import default_loader

class SynapseFolderDataset(BaseDataset):
    """
    DEPRECATED : Only a single counterfactual was generated for each image, 
    so this dataset is not useful for the current experiments.
    """
    def __init__(self, opt, split):
        super().__init__(opt)
        self.augment = (not opt.no_augment)
        self.root = os.path.join(opt.dataroot, split)
        self.original_root = os.path.join(self.root, "original")
        self.counterfactual_root = os.path.join(self.root, "counterfactual")
        self.prediction_root = os.path.join(self.root, "predictions")

        self.original_predictions = pd.read_csv(
            os.path.join(self.prediction_root, "originals.csv"), index_col=0
        )

        imgs, counterfactuals = make_dataset(
            self.original_root, self.counterfactual_root
        )
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in: " + self.root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )
        if len(imgs) != len(counterfactuals):
            raise (
                RuntimeError(
                    "Found {} images in original and {} in counterfactual".format(
                        len(imgs), len(counterfactuals)
                    )
                )
            )
        self.imgs = imgs
        self.counterfactuals = counterfactuals
        
        
        if self.augment and self.split == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    apytorch.transforms.ToTensorV2(),

                ],
                additional_targets={"image1": "image",},

            )
        else:
            self.transform = A.Compose(
            [
                apytorch.transforms.ToTensorV2()
            ],
            additional_targets={"image1": "image",},
        )

        
        # assert False
        # TODO might need to convert to tensor from here
            
    def __len__(self):
        return len(self.imgs)
    

    def __getitem__(self, index):
        path = self.imgs[index]
        counterfactual_path = self.counterfactuals[index]
        image = default_loader(path)/255 *2 -1
        counterfactual = default_loader(counterfactual_path)/255 *2 -1
        # if self.transform is not None:
        transformed = self.transform(image=image, image1=counterfactual)
        # TODO might need to change the names based on expectations from selector
        try :
            transformed["y"] = torch.tensor(np.argmax(self.original_predictions.iloc[index].to_numpy()))
        except Exception as e:
            aux_index = int(self.imgs[index].split("/")[-2].split("_")[0])
            transformed["y"] = torch.tensor(aux_index, dtype=torch.long)
        transformed["y_cf"] = (transformed["y"] + 1)  % 6
        # transformed["x_paths"] = "None"
        # transformed["x_cf_paths"] = "None"
        transformed["x"] = transformed["image"]
        transformed["x_cf"] = transformed["image1"]
        return transformed
    
