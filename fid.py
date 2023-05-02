#### implement fid score of sampled images from a model using pytorch.

"""
imput : sampled images : shape (N,3,H,W)
output : fid score
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg

from torchvision.models.inception import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def get_activations(images, model, batch_size=50, dims=2048, cuda=False, verbose=False):


    