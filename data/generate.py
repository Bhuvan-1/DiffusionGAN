import torch
import torchvision
import numpy as np
import sys


if(len(sys.argv) != 2):
    print("Usage: python generate.py <dataset_name>")
    sys.exit(1)

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print('device : ',device)

DOWNLOAD_ALL = sys.argv[1] == "all"


if(DOWNLOAD_ALL or sys.argv[1] == "cifar10"):
    trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True)

    print("CIFAR10 - (train,test) shapes:", trainset.data.shape, testset.data.shape)
    #CIFAR10 - (train,test) shapes: (50000, 32, 32, 3) (10000, 32, 32, 3)

    np.save("./CIFAR10/CIFAR10_train.npy", trainset.data)
    np.save("./CIFAR10/CIFAR10_test.npy", testset.data)
