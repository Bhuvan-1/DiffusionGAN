import argparse
import torch
from model import LitDiffusionModel, GANDiffusionModel
from dataset import CIFAR10
import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt


seed = 1628
torch.manual_seed(seed)

####  DATA PARAMETERS ####
datapath = './data/CIFAR10/CIFAR10_0_16.npz'
dataset = np.load(datapath)
train_dataset = dataset['train']
train_dataset = torch.from_numpy(train_dataset).permute(0,3,1,2).float()
train_dataset = train_dataset/255



### LOGGING PARAMETERS ###
savedir = './runs/gan'
run_name = "test3"



args = pkl.load(open(savedir+"/"+run_name + "/hparams.pkl",'rb'))


model = GANDiffusionModel(
    down_channels=args['down_channels'],
    up_channels=args['up_channels'],
    n_steps=args['n_steps'],
    lbeta=args['lbeta'],
    ubeta=args['ubeta'],
    noise=args['noise'],
    img_shape=args['img_shape'],
    latent_dim=args['latent_dim'],
)

model.gen = pkl.load(open(savedir+"/"+run_name + "/bestG.pkl",'rb'))
model.disc = pkl.load(open(savedir+"/"+run_name + "/bestD.pkl",'rb'))



images = model.sample(10)


fig, axs = plt.subplots(2,5)
for i in range(2):
    for j in range(5):
        axs[i,j].imshow(images[i*5+j].permute(1,2,0).detach().numpy().clip(0,1) )
        axs[i,j].axis('off')
plt.show()
