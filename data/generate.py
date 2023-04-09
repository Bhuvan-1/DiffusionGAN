import torchvision
import numpy as np
import sys


if(len(sys.argv) != 2):
    print("Usage: python generate.py <dataset_name>")
    sys.exit(1)


DOWNLOAD_ALL = sys.argv[1] == "all"


if(DOWNLOAD_ALL or sys.argv[1] == "cifar10"):
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
