import argparse
import torch
from model import LitDiffusionModel, GANDiffusionModel
from dataset import CIFAR10
import pickle as pkl
import os
import numpy as np


seed = 1628
torch.manual_seed(seed)

####  DATA PARAMETERS ####
train_data = './data/CIFAR10/CIFAR10_0_16.npz'
dataset = np.load(train_data)
train_dataset = dataset['train']
train_dataset = torch.from_numpy(train_dataset).permute(0,3,1,2).float()
train_dataset = train_dataset/255




####  TRAINING PARAMETERS ####
n_epochs = 200
batch_size = 512
lr_g = 5e-4
lr_d = 5e-4



####  MODEL PARAMETERS ####
down_channels = (4,8)
up_channels = (8,4)
n_steps = 8
lbeta = 0.1
ubeta = 20
noise = 'linear'
latent_dim = 128
img_shape = train_dataset.shape[1:]


### LOGGING PARAMETERS ###
savedir = './runs/gan'
run_name = "test"




model = GANDiffusionModel(
    down_channels=down_channels,
    up_channels=up_channels,
    n_steps=n_steps,
    lbeta=lbeta,
    ubeta=ubeta,
    noise=noise,
    img_shape=train_dataset.data.shape[1:],
    latent_dim=latent_dim
)


optimG = torch.optim.SGD(model.gen.parameters(),lr = lr_g)
optimD = torch.optim.SGD(model.disc.parameters(),lr = lr_d)



for epoch in range(n_epochs):
    
        #randomly pick a batch of data
        idx = torch.randperm(train_dataset.size(0))
        batch = train_dataset[idx[:batch_size]]
    
        z_D        = torch.randn( [batch.size(0), model.latent_dim] )
        times_D = torch.randint(1, model.n_steps+1, (batch.size(0),1))

        x_t1_D = model.q_cond_sample(batch , times_D-1)
        x_t_D  = model.q_next_sample(x_t1_D, times_D)
        x_0_D  = model.gen(x_t_D, z_D, times_D)
        x_t1_hat_D = model.q_cond_sample(x_0_D, times_D-1)


        probs_real_D = model.disc(x_t_D, x_t1_D, times_D)
        probs_fake_D = model.disc(x_t_D, x_t1_hat_D, times_D)


        ## training the discriminator
        optimD.zero_grad()
        lossD = ( -torch.log(probs_real_D) - torch.log(1 - probs_fake_D) ).sum()
        lossD.backward()
        optimD.step()



        #second random batch of data
        idx = torch.randperm(train_dataset.size(0))
        batch = train_dataset[idx[:batch_size]]


        ## training the generator
        z_G        = torch.randn( [batch.size(0), model.latent_dim] )
        times_G = torch.randint(1, model.n_steps+1, (batch.size(0),1))

        x_t1_G = model.q_cond_sample(batch, times_G-1)
        x_t_G  = model.q_next_sample(x_t1_G, times_G)
        x_0_G  = model.gen(x_t_G, z_G, times_G)

        probs_fake_G = model.disc(x_t_G, x_0_G, times_G)

        optimG.zero_grad()
        lossG = ( -torch.log(probs_fake_G) ).sum()
        lossG.backward()
        optimG.step()

        # # Regulariser gamma*(gradient of disc)^2
        # grad_disc = torch.autograd.grad(probs_real,x_t1)
        # loss += 0.05*grad_disc**2

        print(f"Epoch {epoch}, : lossD = {lossD}, lossG = {lossG}")

