"""A dataset for synapse data loader from the prototype dataset.

The data is expected to be organized as `{root}/{image_type}/{class_name}/{sample_number}.png` where: 
- `root_dir`
- `image_type` is either `original` or `counterfactual`
- `class_name` is the name of the original class
- `sample_number` is the number of the sample.
"""
import albumentations as A
import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

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


class SynapseFolderDataset(Dataset):
    def __init__(self, root, augment=True):
        self.root = root
        self.original_root = os.path.join(root, "original")
        self.counterfactual_root = os.path.join(root, "counterfactual")
        self.prediction_root = os.path.join(root, "predictions")

        self.original_predictions = pd.read_csv(
            os.path.join(self.prediction_root, "originals.csv"), index_col=0
        )

        imgs, counterfactuals = make_dataset(
            self.original_root, self.counterfactual_root
        )
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in: " + root + "\n"
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
        if augment:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ]
            )
        else:
            self.transform = lambda x: x
        # TODO might need to convert to tensor from here

    def __getitem__(self, index):
        path = self.imgs[index]
        counterfactual_path = self.counterfactuals[index]
        image = default_loader(path)
        counterfactual = default_loader(counterfactual_path)
        if self.transform is not None:
            transformed = self.transform(image=image, xcf=counterfactual)
        # TODO might need to change the names based on expectations from selector
        transformed["y"] = np.argmax(self.original_predictions.iloc[index])
        transformed["ycf"] = (transformed["y"] + 1)  % 6
        transformed["x_path"] = None
        transformed["xcf_path"] = None
        return transformed