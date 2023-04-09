import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt

# A generic dataset object
# Must take the following arguments
# input_dim
# n_classes
class LinearNN(nn.Module):
    def __init__(self, input_dim = 113, output_dim = 1):
        super(LinearNN, self).__init__()
        #self.model = nn.Sequential(
        #        nn.Linear(input_dim, output_dim, bias = True),
        #    )
        self.modellist = [nn.Linear(input_dim, output_dim, bias = True)]
        self.model = nn.Sequential(*self.modellist)

    def forward(self, x):
        ans = self.model(x)
        return ans

class ConvBlocks(nn.Module):
    def __init__(self, input_dim = 64, hidden_dim1 = 128, hidden_dim2 = 256, hidden_dim3 = 512, output_dim = 9, is_downsample=True):
        super(ConvBlocks, self).__init__()
        #self.model = nn.Sequential(
        #        nn.Linear(input_dim, output_dim, bias = True),
        #    )

        self.is_downsample = is_downsample

        self.basic_block1_list = [
            nn.Conv2d(input_dim, hidden_dim1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ]
        self.basic_block1 = nn.Sequential(*self.basic_block1_list)

        self.downsample1_list = [
            nn.Conv2d(input_dim, hidden_dim1, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ]
        self.downsample1 = nn.Sequential(*self.downsample1_list)

        self.basic_block2_list = [
            nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ]
        self.basic_block2 = nn.Sequential(*self.basic_block2_list)


        self.basic_block3_list = [
            nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ]
        self.basic_block3 = nn.Sequential(*self.basic_block3_list)

        self.downsample2_list = [
            nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ]
        self.downsample2 = nn.Sequential(*self.downsample2_list)

        self.basic_block4_list = [
            nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ]
        self.basic_block4 = nn.Sequential(*self.basic_block4_list)


        self.basic_block5_list = [
            nn.Conv2d(hidden_dim2, hidden_dim3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim3, hidden_dim3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ]
        self.basic_block5 = nn.Sequential(*self.basic_block5_list)

        self.downsample3_list = [
            nn.Conv2d(hidden_dim2, hidden_dim3, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(hidden_dim3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ]
        self.downsample3 = nn.Sequential(*self.downsample3_list)

        self.basic_block6_list = [
            nn.Conv2d(hidden_dim3, hidden_dim3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim3, hidden_dim3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ]
        self.basic_block6 = nn.Sequential(*self.basic_block6_list)

        self.adaptive_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.linear = nn.Linear(in_features=hidden_dim3, out_features=output_dim, bias=True)

    def forward(self, x):
        out = self.basic_block1(x)
        if self.is_downsample:
            add_this = self.downsample1(x)
        
        out += add_this

        out = self.basic_block2(out)


        out2 = self.basic_block3(out)
        if self.is_downsample:
            add_this = self.downsample2(out)
        
        out2 += add_this

        out = self.basic_block4(out2)


        out2 = self.basic_block5(out)
        if self.is_downsample:
            add_this = self.downsample3(out)
        
        out2 += add_this

        out = self.basic_block6(out2)


        out = self.adaptive_pooling(out)
 
        d1 = out.shape[0]
        d2 = out.shape[1]
        out = self.linear(out.reshape((d1,d2)))
        return out


class ConvBlocks_CIFAR10(nn.Module):
    def __init__(self, input_dim = 128, hidden_dim1 = 256, hidden_dim2 = 512, output_dim = 10, is_downsample=True):
        super(ConvBlocks_CIFAR10, self).__init__()
        #self.model = nn.Sequential(
        #        nn.Linear(input_dim, output_dim, bias = True),
        #    )

        self.is_downsample = is_downsample

        self.basic_block1_list = [
            nn.Conv2d(input_dim, hidden_dim1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ]
        self.basic_block1 = nn.Sequential(*self.basic_block1_list)

        self.downsample1_list = [
            nn.Conv2d(input_dim, hidden_dim1, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ]
        self.downsample1 = nn.Sequential(*self.downsample1_list)

        self.basic_block2_list = [
            nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ]
        self.basic_block2 = nn.Sequential(*self.basic_block2_list)

        self.basic_block3_list = [
            nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ]
        self.basic_block3 = nn.Sequential(*self.basic_block3_list)

        self.downsample2_list = [
            nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ]
        self.downsample2 = nn.Sequential(*self.downsample2_list)

        self.basic_block4_list = [
            nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ]
        self.basic_block4 = nn.Sequential(*self.basic_block4_list)

        self.adaptive_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.linear = nn.Linear(in_features=hidden_dim2, out_features=output_dim, bias=True)

    def forward(self, x):
        out = self.basic_block1(x)
        if self.is_downsample:
            add_this = self.downsample1(x)
        
        out += add_this

        out = self.basic_block2(out)

        out2 = self.basic_block3(out)
        if self.is_downsample:
            add_this = self.downsample2(out)
        
        out2 += add_this

        out = self.basic_block4(out2)

        out = self.adaptive_pooling(out)
 
        d1 = out.shape[0]
        d2 = out.shape[1]
        out = self.linear(out.reshape((d1,d2)))
        return out