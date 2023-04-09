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

from torch.utils.data.dataset import Dataset

import os
import argparse

import random

from numpy import load

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class FashionMNIST_ROTATE(Dataset):
    def __init__(self, train, transform):
        super(FashionMNIST_ROTATE, self).__init__()
        if train:
            seed_torch(0)
        else:
            seed_torch(1)
        self.train = train
        self.transform = transform
        self.num_classes = 10
        data = load('fashion-mnist.npz')

        if self.train:
            self.data_size = 60000
            train_images = data['train_images']
            train_images = np.repeat(train_images[:,:,:,np.newaxis],3,axis = 3)
            train_labels = data['train_labels']

            train_images = torch.Tensor(train_images)
            train_labels = torch.Tensor(train_labels)

            N = train_images.shape[0]
            dim1 = train_images.shape[1]
            dim3 = train_images.shape[3]
            
            train_images_reshape = torch.zeros((N,dim3,dim1,dim1))
            for i in range(dim3):
                train_images_reshape[:,i,:,:] = train_images[:,:,:,i]
                # train_images_reshape[:,i,:,:] = train_images[:,:,:]

            self.data = train_images_reshape
            # self.data = train_images
            self.labels = train_labels
            self.labels = self.labels.type(torch.LongTensor)
        else:
            self.data_size = 10000
            test_images = data['test_images']
            test_images = np.repeat(test_images[:,:,:,np.newaxis],3,axis = 3)
            test_labels = data['test_labels']

            test_images = torch.Tensor(test_images)
            test_labels = torch.Tensor(test_labels)

            N = test_images.shape[0]
            dim1 = test_images.shape[1]
            dim3 = test_images.shape[3]
            
            test_images_reshape = torch.zeros((N,dim3,dim1,dim1))
            for i in range(dim3):
                test_images_reshape[:,i,:,:] = test_images[:,:,:,i]
                # test_images_reshape[:,i,:,:] = test_images[:,:,:]

            self.data = test_images_reshape
            # self.data = test_images
            self.labels = test_labels
            self.labels = self.labels.type(torch.LongTensor)
        
    def __getitem__(self, i):
        return i, self.data[i], self.labels[i]
        
    def __len__(self):
        return self.data.shape[0]

if(__name__ == '__main__'):

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    transform_normal = transforms.Compose([

        transforms.CenterCrop(21),
        torchvision.transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)),
    ])

    transform_rotate = transforms.Compose([

        transforms.RandomRotation((30,30.00001)),
        transforms.CenterCrop(21),
        torchvision.transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)),
    ])


    trainset_normal = FashionMNIST_ROTATE( train=True, transform=transform_normal)
    trainset_rotate = FashionMNIST_ROTATE( train=True, transform=transform_rotate)

    trainloader_normal = torch.utils.data.DataLoader(
        trainset_normal, batch_size=100, shuffle=True, num_workers=2)
    trainloader_rotate = torch.utils.data.DataLoader(
        trainset_rotate, batch_size=100, shuffle=True, num_workers=2)

    # print(trainset_normal.data.shape)
    # data = trainset_normal.data / 255 # data is numpy array
    # mean = data.mean(axis = (0,2,3)) 
    # std = data.std(axis = (0,2,3))
    # print(f"Mean : {mean}   STD: {std}")
    # #Mean : tensor([0.2860, 0.2860, 0.2860])   STD: tensor([0.3530, 0.3530, 0.3530])


    testset_normal = FashionMNIST_ROTATE( train=False, transform=transform_normal)
    testset_rotate = FashionMNIST_ROTATE( train=False, transform=transform_rotate)
        
    testloader_normal = torch.utils.data.DataLoader(
        testset_normal, batch_size=100, shuffle=True, num_workers=2)
    testloader_rotate = torch.utils.data.DataLoader(
        testset_rotate, batch_size=100, shuffle=True, num_workers=2)


    classes = ('unrotated','rotated')


    net = models.resnet18(pretrained=True)

    for param in net.parameters():
        param.requires_grad = False

    net = net.to(device)
    hidden_size = 64
    dim2 = 7
    n_components = 64

    feature_extractor = torch.nn.Sequential(*list(net.children())[:-5])

    d = {}

    train_features = np.zeros((50000,hidden_size,dim2,dim2))
    train_labels = np.zeros((50000))
    test_features = np.zeros((10000,hidden_size,dim2,dim2))
    test_labels = np.zeros((10000))

    for batch_idx, (_, inputs, targets) in enumerate(testloader_normal):
        if(batch_idx >= 50): break
        inputs = inputs.to(device)
        features = feature_extractor(inputs).squeeze()
        test_features[batch_idx*200:batch_idx*200+100] = features.cpu().numpy()
        test_labels[batch_idx*200:batch_idx*200+100] = np.zeros(100)
    for batch_idx, (_, inputs, targets) in enumerate(testloader_rotate):
        if(batch_idx >= 50): break
        inputs = inputs.to(device)
        features = feature_extractor(inputs).squeeze()
        test_features[batch_idx*200+100:batch_idx*200+200] = features.cpu().numpy()
        test_labels[batch_idx*200+100:batch_idx*200+200] = np.ones(100)

    test_features,test_labels = shuffle(test_features,test_labels,random_state=0)

    d['resnet18_test_features'] = test_features
    d['test_labels'] = test_labels


    for batch_idx, (_, inputs, targets) in enumerate(trainloader_normal):
        if(batch_idx >= 250): break
        inputs = inputs.to(device)
        features = feature_extractor(inputs).squeeze()
        train_features[batch_idx*200:batch_idx*200+100] = features.cpu().numpy()
        train_labels[batch_idx*200:batch_idx*200+100] = np.zeros(100)
    for batch_idx, (_, inputs, targets) in enumerate(trainloader_rotate):
        if(batch_idx >= 250): break 
        inputs = inputs.to(device)
        features = feature_extractor(inputs).squeeze()
        train_features[batch_idx*200+100:batch_idx*200+200] = features.cpu().numpy()
        train_labels[batch_idx*200+100:batch_idx*200+200] = np.ones(100)

    train_features,train_labels = shuffle(train_features,train_labels,random_state=0)

    d['resnet18_train_features'] = train_features
    d['train_labels'] = train_labels


    for k,v in d.items():
        print(k,v.shape)
        
    torch.save(d,'fashionmnist-rotate_without_pca_l5_50K.pth')