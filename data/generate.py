import torch
import torchvision
import numpy as np
import sys
import cv2


if(len(sys.argv) == 1):
    print("Usage: python generate.py <dataset_name> [classes] [new_size]")
    sys.exit(1)

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print('device : ',device)

DOWNLOAD_ALL = sys.argv[1] == "all"

"""
CLASSES OF CIFAR10
0: airplane
1: automobile
2: bird ..........X
3: cat ...........X
4: deer...........X
5: dog
6: frog .........X
7: horse
8: ship ..........X
9: truck

"""
if(DOWNLOAD_ALL or sys.argv[1].lower() == "cifar10"):


    if(len(sys.argv) == 2):
        print("Downloading all classes of CIFAR10")
        l = [0,1,2,3,4,5,6,7,8,9]
        name = "all"
    else:
        name = sys.argv[2]
        l = sys.argv[2].split(',')
        l = [int(i) for i in l]
        print("Classes to be downloaded:", l)


    trainset = torchvision.datasets.CIFAR10(root='./CIFAR10/data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./CIFAR10/data', train=False, download=True)

    #filtering
    trainset.data = trainset.data[np.isin(trainset.targets, l)]
    trainset.targets = np.array(trainset.targets)[np.isin(trainset.targets, l)]

    testset.data = testset.data[np.isin(testset.targets, l)]
    testset.targets = np.array(testset.targets)[np.isin(testset.targets, l)]


    if(len(sys.argv) == 4):
        new_size = int(sys.argv[3])
        print("Resizing images to", new_size, "x", new_size)
        trainset.data = np.array([cv2.resize(img, (new_size, new_size)) for img in trainset.data])
        testset.data = np.array([cv2.resize(img, (new_size, new_size)) for img in testset.data])

        name = name + "_" + str(new_size)
    else:
        name = name + "_32"
    
    print("CIFAR10 - (train,test) shapes:", trainset.data.shape, testset.data.shape)
    #CIFAR10 - (train,test) shapes: (50000, 32, 32, 3) (10000, 32, 32, 3)

    np.savez("./CIFAR10/CIFAR10_" + name + ".npz", train=trainset.data, test=testset.data, train_labels=trainset.targets, test_labels=testset.targets)
