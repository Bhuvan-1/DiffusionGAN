import argparse
import torch
import pytorch_lightning as pl
from model import LitDiffusionModel
from dataset import ThreeDSinDataset

parser = argparse.ArgumentParser()

model_args = parser.add_argument_group('model')
model_args.add_argument('--n_dim', type=int, default=3, help='Number of dimensions')
model_args.add_argument('--n_steps', type=int, default=50, help='Number of diffusion steps')
model_args.add_argument('--lbeta', type=float, default=1e-5, help='Lower bound of beta')
model_args.add_argument('--ubeta', type=float, default=1.28e-2, help='Upper bound of beta')
model_args.add_argument('--n_layers', type=int, default=5, help='Number of Layers in the model')
model_args.add_argument('--embedD', type=int, default=4, help='Time Embedding Dimension')
model_args.add_argument('--embedN', type=int, default=100, help='Time Embedding frequency')
model_args.add_argument('--hidden_size', type=int, default=256, help='Hidden Layer Size of the model')
model_args.add_argument('--noise', type=str, default='linear', help='Noise Schedule')


training_args = parser.add_argument_group('training')
training_args.add_argument('--seed', type=int, default=1618, help='Random seed for experiments')
training_args.add_argument('--n_epochs', type=int, default=500, help='Number of training epochs')
training_args.add_argument('--batch_size', type=int, default=1024, help='Batch size for training dataloader')
training_args.add_argument('--train_data_path', type=str, default='./data/3d_sin_5_5_train.npy', help='Path to training data numpy file')
training_args.add_argument('--savedir', type=str, default='./runs/', help='Root directory where all checkpoint and logs will be saved')

args = parser.parse_args()

n_dim = args.n_dim
n_steps = args.n_steps
lbeta = args.lbeta
ubeta = args.ubeta
n_layers = args.n_layers
embedD = args.embedD
embedN = args.embedN
hidden_size = args.hidden_size
noise = args.noise


pl.seed_everything(args.seed)
batch_size = args.batch_size
n_epochs = args.n_epochs
savedir = args.savedir

litmodel = LitDiffusionModel(
    n_dim=n_dim, 
    n_steps=n_steps, 
    lbeta=lbeta, 
    ubeta=ubeta,
    n_layers=n_layers,
    embedD=embedD,
    embedN=embedN,
    hidden_size=hidden_size,
    noise=noise,
)

train_dataset = ThreeDSinDataset(args.train_data_path)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

run_name = f'n_dim={n_dim},n_steps={n_steps},lbeta={lbeta:.3e},ubeta={ubeta:.3e},batch_size={batch_size},n_epochs={n_epochs}'
# run_name = 'results'

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
