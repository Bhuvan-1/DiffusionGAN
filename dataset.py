import numpy as np
from torch.utils.data import Dataset
import random
import os
import torch
import matplotlib.pyplot as plt


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class CIFAR10(Dataset):
    def __init__(self, npzfile, train = True, mean=None, std=None):
        super().__init__()

        seed_torch(0)

        DICT = np.load('./runs/CIFAR10/' + npzfile)

        if(train):
            self.data = DICT['train']
        else:
            self.data = DICT['test']
       

        self.data = self.data.astype(np.float32)/255.0

        # Convert from H x W x C to C x H x W
        self.data = self.data.transpose(0, 3, 1, 2)
        print("DataSet Shape : ",self.data.shape)

        if mean is None:
            mean = np.mean(self.data, axis=0)
        if std is None:
            std = np.std(self.data, axis=0)

        self.mean = mean
        self.std = std
        
        self.data = (self.data - mean) / std

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

