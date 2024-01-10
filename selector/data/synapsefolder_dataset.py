"""A dataset for synapse data loader from the prototype dataset.

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


def default_loader(path):
    return np.array(Image.open(path))[
        ..., 0
    ]  # It's a grayscale image, so we only need one channel


class SynapseFolderDataset(BaseDataset):
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
        
        
        if self.augment:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    apytorch.transforms.ToTensorV2(),

                ]
            )
        else:
            self.transform = A.Compose(
            [
                apytorch.transforms.ToTensorV2()
            ]
        )

        
        # assert False
        # TODO might need to convert to tensor from here
            
    def __len__(self):
        return len(self.imgs)
    

    def __getitem__(self, index):
        path = self.imgs[index]
        counterfactual_path = self.counterfactuals[index]
        image = default_loader(path)
        counterfactual = default_loader(counterfactual_path)
        # if self.transform is not None:
        transformed = self.transform(image=image, xcf=counterfactual)
        print(transformed)
        # TODO might need to change the names based on expectations from selector

        transformed["y"] = torch.from_numpy(np.argmax(self.original_predictions.iloc[index]))
        transformed["y_cf"] = torch.from_numpy((transformed["y"] + 1)  % 6)
        # transformed["x_paths"] = "None"
        # transformed["x_cf_paths"] = "None"
        transformed["x"] = transformed["image"]
        transformed["x_cf"] = transformed["xcf"]
        return transformed
    
