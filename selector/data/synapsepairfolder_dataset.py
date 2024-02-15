"""A dataset for synapse data loader using counterfactuals generated with Gans.
Each pair is considered one example.
"""

import logging
import numpy as np
from .base_dataset import BaseDataset
from .utils_pair_image_folder import make_dataset, find_classes, is_image_file, default_loader
import albumentations as A
import albumentations.pytorch as apytorch
import os
import torch


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)




class SynapsePairFolderDataset(BaseDataset,): 

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
        parser.add_argument("--bypass_counterfactual", action="store_true", help="If true, we will only return the path to the original \
                            image and dummy counterfactuals. Useful in test.")
        return parser
    

    def __init__(self, opt, split="train"):
        """A dataset that loads images from paired directories, where one has images 
        generated based on the other. In this case, we consider that one example is
        a pair of images, one from the source directory and the other from the paired directory.
        This is to use in collaboration with the Pathwise Selector Model directly.

        Source directory is expected to be of the form: 
        ```
        directory/
        ├── class_x
        │   ├── xxx.ext
        │   ├── xxy.ext
        │   └── xxz.ext
        └── class_y
            ├── 123.ext
            ├── nsdf3.ext
            └── ...
            └── asd932_.ext
        ```
            
        Paired directory should be:
        ```
        directory/
        ├── class_x
        |   └── class_y
        │       ├── xxx.ext
        │       ├── xxy.ext
        │       └── xxz.ext
        └── class_y
            └── class_x
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext
        ```
        Note that this will not work if the file names do not match!
        """
        super().__init__(opt)

        source_directory = os.path.join(opt.dataroot, split)
        paired_directory = os.path.join(opt.dataroot_counterfactual, split)       
        classes, class_to_idx = find_classes(source_directory)
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples, _ = make_dataset(source_directory, paired_directory, class_to_idx, is_valid_file=is_image_file, bypass_counterfactual=opt.bypass_counterfactual)
        # self.transform = transform
        if self.augment and self.split == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    apytorch.transforms.ToTensorV2(),

                ],
                additional_targets={"imagecf": "image",},
            )
        else:
            self.transform = A.Compose(
            [
                apytorch.transforms.ToTensorV2()
            ],
            additional_targets={"imagecf": "image",},

        )


    def __getitem__(self, index):
        path, target_path, class_index, target_class_index = self.samples[index]
        sample = default_loader(path)
        if self.opt.bypass_counterfactual:
            target_sample = np.zeros_like(sample)
        else :
            target_sample = default_loader(target_path)
        transformed = self.transform(image=sample, imagecf=target_sample)
        # assert False
        output = {
            "x_path": path,
            "x_cf_path": target_path,
            "x": transformed["image"].to(torch.float32),
            "x_cf": transformed["imagecf"].to(torch.float32),
            "y": torch.tensor(class_index,),
            "y_cf": torch.tensor(target_class_index,),
        }
        return output

    def __len__(self):
        return len(self.samples)


