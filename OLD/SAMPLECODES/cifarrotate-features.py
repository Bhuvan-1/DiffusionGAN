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
from sklearn.utils import shuffle

import os
import argparse
import random

from cifar10_models.resnet import resnet18

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print("device : ", device)

    seed_torch(2)

    transform_normal = transforms.Compose([

        transforms.CenterCrop(24),
        torchvision.transforms.Resize((32,32)),
        # torchvision.transforms.GaussianBlur(kernel_size = 5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
    ])

    transform_rotate = transforms.Compose([

        transforms.RandomRotation((30,30.1)),
        transforms.CenterCrop(24),
        torchvision.transforms.Resize((32,32)),
        # torchvision.transforms.GaussianBlur(kernel_size = 5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
    ])

    trainset_normal = torchvision.datasets.CIFAR10(
        root='./data_cifar', train=True, download=True, transform=transform_normal)
    trainset_rotate = torchvision.datasets.CIFAR10(
        root='./data_cifar', train=True, download=True, transform=transform_rotate)

    trainloader_normal = torch.utils.data.DataLoader(
        trainset_normal, batch_size=100, shuffle=True, num_workers=2)
    trainloader_rotate = torch.utils.data.DataLoader(
        trainset_rotate, batch_size=100, shuffle=True, num_workers=2)

    # #print(trainset_normal.data.shape)
    # data = trainset_normal.data / 255 # data is numpy array
    # mean = data.mean(axis = (0,1,2)) 
    # std = data.std(axis = (0,1,2))
    # print(f"Mean : {mean}   STD: {std}")
    # #Mean : [0.491 0.482 0.447]   STD: [0.247 0.243 0.262]


    testset_normal = torchvision.datasets.CIFAR10(
        root='./data_cifar', train=False, download=True, transform=transform_normal)
    testset_rotate = torchvision.datasets.CIFAR10(
        root='./data_cifar', train=False, download=True, transform=transform_rotate)
        
    testloader_normal = torch.utils.data.DataLoader(
        testset_normal, batch_size=100, shuffle=True, num_workers=2)
    testloader_rotate = torch.utils.data.DataLoader(
        testset_rotate, batch_size=100, shuffle=True, num_workers=2)


    classes = ('unrotated','rotated')


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

    # #_______________-Only 5classes case____________
    # train_features = np.zeros((50000*2,hidden_size,dim2,dim2))
    # train_labels = np.zeros((50000*2))
    # train_original_labels = np.zeros((50000*2))

    # test_features = np.zeros((10000*2,hidden_size,dim2,dim2))
    # test_labels = np.zeros((10000*2))
    # test_original_labels = np.zeros((10000*2))

    #__________10classes....50% random examplescase________
    train_features = np.zeros((50000,hidden_size,dim2,dim2))
    train_labels = np.zeros((50000))

    test_features = np.zeros((10000,hidden_size,dim2,dim2))
    test_labels = np.zeros((10000))

    for batch_idx, (inputs, targets) in enumerate(testloader_normal):
        if(batch_idx >= 50): break    #__________10C...50%examples case.
        inputs = inputs.to(device)
        features = feature_extractor(inputs).squeeze()
        test_features[batch_idx*200:batch_idx*200+100] = features.cpu().numpy()
        test_labels[batch_idx*200:batch_idx*200+100] = np.zeros(100)
        # test_original_labels[batch_idx*200:batch_idx*200+100] = targets.squeeze().cpu().numpy() #.....5C case
    for batch_idx, (inputs, targets) in enumerate(testloader_rotate):
        if(batch_idx >= 50): break   #__________10C...50%examples case.
        inputs = inputs.to(device)
        features = feature_extractor(inputs).squeeze()
        test_features[batch_idx*200+100:batch_idx*200+200] = features.cpu().numpy()
        test_labels[batch_idx*200+100:batch_idx*200+200] = np.ones(100)
        # test_original_labels[batch_idx*200+100:batch_idx*200+200] = targets.squeeze().cpu().numpy() #...5C case.

    # #________5C case ________
    # test_features = test_features[test_original_labels < 5]
    # test_labels = test_labels[test_original_labels < 5]

    test_features,test_labels = shuffle(test_features,test_labels,random_state=0)

    d['resnet18_test_features'] = test_features
    d['test_labels'] = test_labels


    for batch_idx, (inputs, targets) in enumerate(trainloader_normal):
        if(batch_idx >= 125): break   #________10C 50%examples case.
        inputs = inputs.to(device)
        features = feature_extractor(inputs).squeeze()
        train_features[batch_idx*200:batch_idx*200+100] = features.cpu().numpy()
        train_labels[batch_idx*200:batch_idx*200+100] = np.zeros(100)
        # train_original_labels[batch_idx*200:batch_idx*200+100] = targets.squeeze().cpu().numpy()
    for batch_idx, (inputs, targets) in enumerate(trainloader_rotate):
        if(batch_idx >= 125): break    #________10C 50%examples case.
        inputs = inputs.to(device)
        features = feature_extractor(inputs).squeeze()
        train_features[batch_idx*200+100:batch_idx*200+200] = features.cpu().numpy()
        train_labels[batch_idx*200+100:batch_idx*200+200] = np.ones(100)
        # train_original_labels[batch_idx*200+100:batch_idx*200+200] = targets.squeeze().cpu().numpy()


    # train_features = train_features[train_original_labels < 5]
    # train_labels = train_labels[train_original_labels < 5]

    train_features,train_labels = shuffle(train_features,train_labels,random_state=0)

    d['resnet18_train_features'] = train_features
    d['train_labels'] = train_labels


    for k,v in d.items():
        print(k,v.shape)

        
    torch.save(d,'cifar-rotate_without_pca_l4_10C_half_2.pth')