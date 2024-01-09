import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision.datasets as datasets
import random
import torch

class MnistDuckDataset(BaseDataset):
    """This dataset does not make any sense. It's just a creation for me to understand how the data is passed to the model.
    What I do is I pair randomly an image from the MNIST dataset with a given target to a another image from the MNIST dataset with a different target.
    """

    def __init__(self, opt, train_dataset=True):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.mnist = datasets.MNIST(root=opt.dataroot, train=train_dataset, download=True, transform=None,)
        self.input_nc = 1
        self.corresponding_index = {}

        


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains x, x_cf, x_paths and x_cf_paths
            x (tensor) - - an image in the input domain
            x_cf (tensor) - - its corresponding image in the target domain
            x_paths (str) - - image paths
            x_cf_paths (str) - - image paths (same as x_paths)
            y (tensor) - - target from image
            y_cf (tensor) - - its corresponding image in the target domain
        """

        x, y = self.mnist.__getitem__(index)
        if y not in self.corresponding_index:
            y_cf = y
            while y_cf == y:
                index = random.randint(0, len(self.mnist)-1)
                x_cf, y_cf = self.mnist.__getitem__(index)
            self.corresponding_index[y] = index
        else:
            index = self.corresponding_index[y]
            x_cf, y_cf = self.mnist.__getitem__(index)


        transform_params = get_params(self.opt, x.size)

        x_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        x_cf_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))

        x = x_transform(x).reshape(1, 28, 28)
        x_cf = x_cf_transform(x_cf).reshape(1, 28, 28)

        y = torch.tensor(y)
        y_cf = torch.tensor(y_cf)

        return {'x': x, 'x_cf': x_cf, 'x_paths': index, 'x_cf_paths': index, 'y': y, 'y_cf': y_cf}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.mnist)
