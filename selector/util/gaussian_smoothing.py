from math import ceil

import torch
from torch.nn.functional import conv2d
from torch.distributions import Normal


def gaussian_kernel_1d(sigma: float, num_sigmas: float = 3.) -> torch.Tensor:
    
    radius = ceil(num_sigmas * sigma)
    support = torch.arange(-radius, radius + 1, dtype=torch.float)
    kernel = Normal(loc=0, scale=sigma).log_prob(support).exp_()
    # Ensure kernel weights sum to 1, so that image brightness is not altered
    return kernel.mul_(1 / kernel.sum())


def gaussian_filter_2d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    
    kernel_1d = gaussian_kernel_1d(sigma).to(img.device)  # Create 1D Gaussian kernel
    
    padding = len(kernel_1d) // 2  # Ensure that image size does not change
    # img = img.unsqueeze(0).unsqueeze_(0)  # Need 4D data for ``conv2d()``
    # Convolve along columns and rows
    img = conv2d(img, weight=kernel_1d.view(1, 1, -1, 1), padding=(padding, 0))
    img = conv2d(img, weight=kernel_1d.view(1, 1, 1, -1), padding=(0, padding))
    return img # Make 2D again
