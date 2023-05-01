import argparse
import torch
import pytorch_lightning as pl
from model import LitDiffusionModel, GANDiffusionModel
from dataset import CIFAR10

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gan')

args = parser.parse_args()


seed = 1628
train_data = 'CIFAR10_0_16.npz'
batch_size = 512


pl.seed_everything(seed)

train_dataset = CIFAR10(train_data)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

if(args.model == 'ddpm'):

    n_steps = 500
    lbeta = 1e-4
    ubeta = 0.02
    noise = 'linear'

    n_epochs = 100
    savedir = './runs/ddpm'

    down_channels = (64,128)
    up_channels = (128,64)

    litmodel = LitDiffusionModel(
        n_steps=n_steps, 
        lbeta=lbeta, 
        ubeta=ubeta,
        noise=noise,
        down_channels = down_channels,
        up_channels = up_channels,
        img_shape=train_dataset.data.shape[1:]
    )


    run_name = "N=500,100ep,1e-4,0.02,64_128"

if(args.model == 'gan'):

    n_steps = 8
    lbeta = 0.1
    ubeta = 20
    noise = 'linear'

    n_epochs = 200
    savedir = './runs/gan'

    down_channels = (16,32)
    up_channels = (32,16)

    litmodel = GANDiffusionModel(
        down_channels = down_channels,
        up_channels = up_channels,
        img_shape=train_dataset.data.shape[1:]
    )

    run_name = "test"


trainer = pl.Trainer(
    deterministic=True,
    logger=pl.loggers.TensorBoardLogger(f'{savedir}/{run_name}/'),
    max_epochs=n_epochs,
    log_every_n_steps=1,
    callbacks=[
        # A dummy model checkpoint callback that stores the latest model at the end of every epoch
        pl.callbacks.ModelCheckpoint(
            dirpath=f'{savedir}/{run_name}/',
            filename='{epoch:04d}-{train_loss:.3f}',
            save_top_k=1,
            monitor='epoch',
            mode='max',
            save_last=True,
            every_n_epochs=1,
        ),
    ]
)

trainer.fit(model=litmodel, train_dataloaders=train_dataloader)
