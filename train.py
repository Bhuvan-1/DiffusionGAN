import argparse
import torch
import pytorch_lightning as pl
from model import LitDiffusionModel
from dataset import CIFAR10

parser = argparse.ArgumentParser()

n_steps = 300
lbeta = 1e-5
ubeta = 1.28e-2
noise = 'linear'

seed = 1628
n_epochs = 100
batch_size = 512
train_data = 'CIFAR10_0_16.npz'
savedir = './runs/'

down_channels = (4,8,16)
up_channels = (16,8,4)
down_channel_str = '_'.join([str(x) for x in down_channels])
up_channel_str = '_'.join([str(x) for x in up_channels])



pl.seed_everything(seed)

train_dataset = CIFAR10(train_data)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

litmodel = LitDiffusionModel(
    n_steps=n_steps, 
    lbeta=lbeta, 
    ubeta=ubeta,
    noise=noise,
    down_channels = down_channels,
    up_channels = up_channels,
    img_shape=train_dataset.data.shape[1:]
)


run_name = f'data={train_data},n_steps={n_steps},lbeta={lbeta:.3e},ubeta={ubeta:.3e},batch_size={batch_size},n_epochs={n_epochs},noise={noise}'


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
