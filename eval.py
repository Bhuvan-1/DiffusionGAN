import os
import argparse
import torch
import numpy as np
from myModels import LitDiffusionModel
from eval_utils import *

parser = argparse.ArgumentParser()

model_args = parser.add_argument_group('model')
model_args.add_argument('--ckpt_path', type=str, help='Path to the model checkpoint', required=True)
model_args.add_argument('--hparams_path', type=str, help='Path to model hyperparameters', required=True)

data_args = parser.add_argument_group('data')
data_args.add_argument('--train_data_path', type=str, default='./data/CIFAR10/CIFAR10_0.npz', help='Path to training data numpy file')
data_args.add_argument('--test_data_path', type=str, default='./data/3d_sin_5_5_test.npy', help='Path to test data numpy file')

eval_args = parser.add_argument_group('evaluation')
eval_args.add_argument('--savedir', type=str, default='./results/', help='Path to directory for saving evaluation results')
eval_args.add_argument('--n_runs', type=int, default=3, help='Number of runs of evaluation')
eval_args.add_argument('--eval_emd', action='store_true', help='Calculate Earth Mover\'s Distance')
eval_args.add_argument('--eval_emd_samples', type=int, default=128, help='Number of random samples to be sampled for calculating EMD')
eval_args.add_argument('--eval_nll', action='store_true', help='Calculate negative log likelihood')
eval_args.add_argument('--eval_chamfer', action='store_true', help='Calculate Chamfer Distance (using `chamferdist`)')
args = parser.parse_args()

litmodel = LitDiffusionModel.load_from_checkpoint(
    args.ckpt_path, 
    hparams_file=args.hparams_path
)
litmodel.eval()

traindata = np.load(args.train_data_path)
testdata = np.load(args.test_data_path)

mean = np.mean(traindata, axis=0)
std = np.std(traindata, axis=0)

traindata = (traindata - mean) / std
testdata = (testdata - mean) / std

traindata = torch.from_numpy(traindata)
testdata = torch.from_numpy(testdata)

os.makedirs(args.savedir, exist_ok=True)

for i_run in range(args.n_runs):
    print(64*'-')
    print(f'Evaluation run {i_run+1}/{args.n_runs}')
    print(64*'-')
    with torch.no_grad():
        gendata, intermediate = litmodel.sample(testdata.size(0), progress=True, return_intermediate=True)

    with open(f'{args.savedir}/{i_run:02d}_log.txt', 'w') as f:
        f.write('Results\n')
    # EMD
    if args.eval_emd:
        idx = np.random.choice(np.arange(gendata.size(0)), size=args.eval_emd_samples, replace=False)
        test_emd = get_emd(testdata[idx].numpy(), gendata[idx].numpy())
        train_emd = get_emd(traindata[idx].numpy(), gendata[idx].numpy())
        print(f'test_emd: {test_emd}')
        print(f'train_emd: {train_emd}')
        with open(f'{args.savedir}/{i_run:02d}_log.txt', 'a') as f:
            f.write(f'test_emd: {test_emd}\n')
            f.write(f'train_emd: {train_emd}\n')

    # NLL
    if args.eval_nll:
        test_nll = get_nll(testdata, gendata).item()
        train_nll = get_nll(traindata, gendata).item()
        print(f'test_nll: {test_nll}')
        print(f'train_nll: {train_nll}')
        with open(f'{args.savedir}/{i_run:02d}_log.txt', 'a') as f:
            f.write(f'test_nll: {test_nll}\n')
            f.write(f'train_nll: {train_nll}\n')

    # Chamfer
    if args.eval_chamfer:
        from chamferdist import ChamferDistance
        cd = ChamferDistance()
        test_chamfer = cd(
            testdata.unsqueeze(0).float(), 
            gendata.unsqueeze(0).float()
        ).item()
        train_chamfer = cd(
            traindata.unsqueeze(0).float(),
            gendata.unsqueeze(0).float()
        ).item()
        print(f'test_chamfer: {test_chamfer}')
        print(f'train_chamfer: {train_chamfer}')
        with open(f'{args.savedir}/{i_run:02d}_log.txt', 'a') as f:
            f.write(f'test_chamfer: {test_chamfer}\n')
            f.write(f'train_chamfer: {train_chamfer}\n')
    
    print(64*'-')
