import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os
import argparse

from cifar10_models.resnet import resnet18

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print('device : ',device)

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data_cifar', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data_cifar', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


net = resnet18(pretrained=True)
# net = models.resnet18(pretrained=True)

for param in net.parameters():
    param.requires_grad = False

net = net.to(device)
hidden_size = 128
dim2 = 8
n_components = 8192

feature_extractor = torch.nn.Sequential(*list(net.children())[:-4])

d = {}

train_features = np.zeros((50000,hidden_size,dim2,dim2))
train_labels = np.zeros((50000))
test_features = np.zeros((10000,hidden_size,dim2,dim2))
test_labels = np.zeros((10000))

for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs = inputs.to(device)
    features = feature_extractor(inputs).squeeze()
    test_features[batch_idx*100:batch_idx*100+100] = features.cpu().numpy()
    test_labels[batch_idx*100:batch_idx*100+100] = targets.squeeze().cpu().numpy()

d['resnet18_test_features'] = test_features
d['test_labels'] = test_labels


for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs = inputs.to(device)
    features = feature_extractor(inputs).squeeze()
    train_features[batch_idx*100:batch_idx*100+100] = features.cpu().numpy()
    train_labels[batch_idx*100:batch_idx*100+100] = targets.squeeze().cpu().numpy()

d['resnet18_train_features'] = train_features
d['train_labels'] = train_labels



for k,v in d.items():
    print(k,v.shape)
    
torch.save(d,'cifar_without_pca_l4.pth')