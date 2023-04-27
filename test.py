import numpy as np
from model import LitDiffusionModel
import matplotlib.pyplot as plt

DICT = np.load('data/CIFAR10/CIFAR10_0_16.npz')
trainset = DICT['train']/255
mean = trainset.mean(axis=0)
std = trainset.std(axis=0)



litmodel = LitDiffusionModel.load_from_checkpoint(
    './runs/N=200,100ep,64_128/last.ckpt',
    hparams_path = './runs/N=200,100ep,64_128/lightning_logs/version_0/hparams.yaml'
)
litmodel.eval()

X = litmodel.sample(10,return_intermediate=False)

fig, ax = plt.subplots(1,10)
for i in range(10):
    x = X[i].detach().numpy().transpose(1,2,0)*std + mean 
    x = np.clip(x,0,1)
    ax[i].imshow(x)
    ax[i].axis('off')

#save the figure
plt.savefig('sample.png')

