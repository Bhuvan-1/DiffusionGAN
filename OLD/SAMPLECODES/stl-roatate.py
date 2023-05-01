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

from stl10_models.resnet import resnet18

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
        transforms.CenterCrop(64),
        torchvision.transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize( mean = [0.4472, 0.437, 0.405] ,std = [0.2605,0.2566, 0.270]),
    ])

    transform_rotate = transforms.Compose([
        transforms.RandomRotation((30,30.001)),
        transforms.CenterCrop(64),
        torchvision.transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize( mean = [0.4472, 0.437, 0.405] ,std = [0.2605,0.2566, 0.270]),
    ])

    trainset_normal = torchvision.datasets.STL10(
        root='./data_stl', split = 'test', download=True, transform=transform_normal)
    trainset_rotate = torchvision.datasets.STL10(
        root='./data_stl', split = 'test', download=True, transform=transform_rotate)

    # trainset_normal.data = trainset_normal.data.transpose(0,2,3,1)
    # trainset_rotate.data = trainset_rotate.data.transpose(0,2,3,1)

    trainloader_normal = torch.utils.data.DataLoader(
        trainset_normal, batch_size=100, shuffle=True, num_workers=2)
    trainloader_rotate = torch.utils.data.DataLoader(
        trainset_rotate, batch_size=100, shuffle=True, num_workers=2)


    testset_normal = torchvision.datasets.STL10(
        root='./data_stl', split='train', download=True, transform=transform_normal)
    testset_rotate = torchvision.datasets.STL10(
        root='./data_stl', split='train', download=True, transform=transform_rotate)

    # testset_normal.data = testset_normal.data.transpose(0,2,3,1)
    # testset_rotate.data = testset_rotate.data.transpose(0,2,3,1)
     
    testloader_normal = torch.utils.data.DataLoader(
        testset_normal, batch_size=100, shuffle=True, num_workers=2)
    testloader_rotate = torch.utils.data.DataLoader(
        testset_rotate, batch_size=100, shuffle=True, num_workers=2)

    # print(trainset_normal.data.shape)
    # print(testset_normal.data.shape)
    # data = trainset_normal.data / 255 # data is numpy array
    # mean = data.mean(axis = (0,1,2)) 
    # std = data.std(axis = (0,1,2))
    # print(f"Mean : {mean}   STD: {std}")
    # #Mean : [0.44723063 0.43964247 0.40495725]   STD: [0.2605645  0.25666146 0.26997382]


    classes = ('unrotated','rotated')

    net = resnet18(pretrained=True)

    for param in net.parameters():
        param.requires_grad = False

    net = net.to(device)
    hidden_size = 128
    dim2 = 8

    feature_extractor = torch.nn.Sequential(*list(net.children())[:-4])

    d = {}


    ###!!! STL-10 dataset has 5K train, 8K test examples.
    # I have used thoe 8K for training and 5K for testing.

    train_features = np.zeros((16000*2,hidden_size,dim2,dim2))
    train_labels = np.zeros((16000*2))
    test_features = np.zeros((10000,hidden_size,dim2,dim2))
    test_labels = np.zeros((10000))


    for batch_idx, (inputs, targets) in enumerate(testloader_normal):
        inputs = inputs.to(device)
        features = feature_extractor(inputs).squeeze()
        test_features[batch_idx*200:batch_idx*200+100] = features.cpu().numpy()
        test_labels[batch_idx*200:batch_idx*200+100] = np.zeros(100)
    for batch_idx, (inputs, targets) in enumerate(testloader_rotate):
        inputs = inputs.to(device)
        features = feature_extractor(inputs).squeeze()
        test_features[batch_idx*200+100:batch_idx*200+200] = features.cpu().numpy()
        test_labels[batch_idx*200+100:batch_idx*200+200] = np.ones(100)

    test_features,test_labels = shuffle(test_features,test_labels,random_state=0)

    d['resnet18_test_features'] = test_features
    d['test_labels'] = test_labels


    for batch_idx, (inputs, targets) in enumerate(trainloader_normal):
        inputs = inputs.to(device)
        features = feature_extractor(inputs).squeeze()
        train_features[batch_idx*200:batch_idx*200+100] = features.cpu().numpy()
        train_labels[batch_idx*200:batch_idx*200+100] = np.zeros(100)
    for batch_idx, (inputs, targets) in enumerate(trainloader_rotate):
        inputs = inputs.to(device)
        features = feature_extractor(inputs).squeeze()
        train_features[batch_idx*200+100:batch_idx*200+200] = features.cpu().numpy()
        train_labels[batch_idx*200+100:batch_idx*200+200] = np.ones(100)

    train_features,train_labels = shuffle(train_features,train_labels,random_state=0)

    d['resnet18_train_features'] = train_features
    d['train_labels'] = train_labels


    for k,v in d.items():
        print(k,v.shape)


    torch.save(d,'stl_l4_32_10K_half.pth')