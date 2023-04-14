""""
UNET ARCHITECTURE FOR DDPM
input image shape (32,32,3)
output scalar

"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import math


class Block(nn.Module):
    """ 
    Input shape : (N, in_ch, H, W)  or  (N, in_ch*2, H, W, T) in up sampling bcz of skip connection
    Output shape: (N, out_ch, H, W)
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.time_emb_dim = time_emb_dim

        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        h = self.bnorm1(self.relu(self.conv1(x)))

        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]         # Extend last 2 dimensions

        h = h + time_emb

        h = self.bnorm2(self.relu(self.conv2(h)))

        h = self.transform(h)
        return h

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Input shape : (N,1)
    Output shape: (N,D)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        N = 10000
        D = self.dim

        if(len(time.shape) == 1):
            time = time.reshape(-1,1)
        
        # #if time is a torch tensor, convert to scalar.
        # if isinstance(time, torch.Tensor):
        #     t = time.item()
        # else:
        #     t = time

        num = time.shape[0]

        sins = torch.zeros(num,D)
        coss = torch.zeros(num,D)
        embed = torch.zeros(num,D)

        denoms = torch.arange(0, D)
        denoms = -torch.where((denoms%2==0), denoms, denoms-1)/D
        denoms = (N**denoms)

        denoms = denoms.reshape(1,-1)

        sins = torch.sin(time*denoms)
        coss = torch.cos(time*denoms)

        embed[:,0::2] = sins[:,0::2]
        embed[:,1::2] = coss[:,1::2]

        return embed     

class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    Input shape : (N, 3, H, W)
    Output shape: (N, 3, H, W)
    """
    def __init__(self,image_channels = 3, down_channels = (64, 128, 256, 512, 1024), up_channels = (1024, 512, 256, 128, 64), out_dim = 1, time_emb_dim = 32):
        super().__init__()

        self.image_channels = image_channels
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.out_dim = out_dim
        self.time_emb_dim = time_emb_dim


        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([ Block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels)-1)])
        
        # Upsample
        self.ups = nn.ModuleList([ Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], 3, out_dim)

    def forward(self, x, timestep):

        t = self.time_mlp(timestep)
        x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels [APPEND]
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)


class LitDiffusionModel(pl.LightningModule):
    def __init__(self,img_shape = (3,32,32), n_steps=200, lbeta=1e-5, ubeta=1e-2,noise='linear',down_channels = (64, 128, 256, 512, 1024), up_channels = (1024, 512, 256, 128, 64), out_dim = 1, time_emb_dim = 32):
        super().__init__()
        """
        If you include more hyperparams (e.g. `n_layers`), be sure to add that to `argparse` from `train.py`.
        Also, manually make sure that this new hyperparameter is being saved in `hparams.yaml`.
        """
        self.save_hyperparameters()


        self.model = SimpleUnet(img_shape[0],down_channels, up_channels, out_dim, time_emb_dim)
        
        """
        Be sure to save at least these 2 parameters in the model instance.
        """
        self.img_shape = img_shape
        self.lbeta = lbeta
        self.ubeta = ubeta
        self.n_steps = n_steps
        self.noise = noise

        """
        Sets up variables for noise schedule
        """
        self.init_alpha_beta_schedule(lbeta, ubeta)

    def forward(self, x, t):
        """
            x --> shape (N, C, H, W)'
            t --> shape (N, 1)

            Returns: (N, C, H, W)
        """

        if(len(x.shape) == 3):
            x = x[None,:,:,:]
        if(isinstance(t, torch.Tensor)):
            if(len(t.shape) == 1):
                t = t[None,:]
        else:
            t = torch.tensor(t).reshape(1,1)

        return self.model(x, t)

    def init_alpha_beta_schedule(self, lbeta, ubeta):
        """
        Sets up variables for noise schedule
        switch between linear, cosine, and sigmoid noise schedules using the `noise` parameter.

        """
        if( self.noise == 'linear'):
            self.beta = torch.linspace(lbeta, ubeta, self.n_steps)
            self.alpha =  1 - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        elif( self.noise == 'cosine' ):
            times = torch.arange(self.n_steps)/self.n_steps
            self.beta = 1e-5 + (1 - torch.cos(times*torch.pi/2))*0.01
            self.alpha =  1 - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        elif( self.noise == 'sigmoid' ):
            betas = torch.linspace(-6, 6, self.n_steps)
            self.beta = torch.sigmoid(betas) * (ubeta - lbeta) + lbeta
            self.alpha =  1 - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        elif( self.noise == 'cosine_alpha' ):
            s = 0.008
            steps = self.n_steps + 1
            x = torch.linspace(0, self.n_steps , steps)
            alphas_cumprod = torch.cos(((x / self.n_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.beta = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.beta = torch.clip(self.beta, 0.0001, 0.9999)
            self.alpha =  1 - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def training_step(self, batch, batch_idx):
        """
        Implements one training step.
        Given a batch of samples (n_samples, C = 3, H, W) from the distribution you must calculate the loss
        for this batch. Simply return this loss from this function so that PyTorch Lightning will 
        automatically do the backprop for you. 
        Refer to the DDPM paper [1] for more details about equations that you need to implement for
        calculating loss. Make sure that all the operations preserve gradients for proper backprop.
        Refer to PyTorch Lightning documentation [2,3] for more details about how the automatic backprop 
        will update the parameters based on the loss you return from this function.

        References:
        [1]: https://arxiv.org/abs/2006.11239
        [2]: https://pytorch-lightning.readthedocs.io/en/stable/
        [3]: https://www.pytorchlightning.ai/tutorials
        """ 

        epsilons = torch.randn( [batch.size(0)] + list(self.img_shape) )
        times = torch.randint(1, self.n_steps+1, (batch.size(0),1))

        # print(batch.shape)
        # print(epsilons.shape)
        # print(self.alpha_bar[times-1].shape)

        reparams = torch.sqrt(self.alpha_bar[times-1])[:,:,None,None]*batch + torch.sqrt(1 - self.alpha_bar[times-1])[:,:,None,None]*epsilons
        eps_thetas = self.forward(reparams, times)
        loss = torch.sum((epsilons - eps_thetas)**2)

        return loss

    def sample(self, n_samples, progress=False, return_intermediate=False):
        """
        Implements inference step for the DDPM.
        `progress` is an optional flag to implement -- it should just show the current step in diffusion
        reverse process.
        If `return_intermediate` is `False`,
            the function returns a `n_samples` sampled from the learned DDPM
            i.e. a Tensor of size (n_samples, <imag_shape>).
        Else
            the function returns all the intermediate steps in the diffusion process as well 
            i.e. a Tensor of size (n_samples, n_dim) and a list of `self.n_steps` Tensors of size (n_samples, <img_shape>) each.
            Return: (n_samples, <img_shape>)(final result), [(n_samples, <img_shape>)(intermediate) x n_steps]
        """

        X = torch.zeros( [n_samples , self.n_steps+1] + list(self.img_shape) )
        X[:,-1,:] = torch.randn( [n_samples] + list(self.img_shape) )

        self.sigma = torch.sqrt(self.beta)
        sigma = self.sigma
        alpha_bar = self.alpha_bar
        alpha = self.alpha

        for t in range(self.n_steps, 0, -1):

            times = torch.LongTensor([t]).expand(n_samples).reshape(n_samples,1).to(torch.float64)

            X[:,t-1] =  (1/torch.sqrt(alpha[t-1]))*( X[:,t] - self.forward(X[:,t], times) * ( (1 - alpha[t-1]) / torch.sqrt(1 - alpha_bar[t-1]) ) )
            if(t > 1):
                X[:,t-1] = X[:,t-1] + torch.randn( [n_samples] + list(self.img_shape) ) * sigma[t-1]
        
        if(not return_intermediate):
            return X[:,0,:]
        else:
            return ( X[:,0], [ X[:,i] for i in range(self.n_steps,0,-1) ] )

    def configure_optimizers(self):
        """
        Sets up the optimizer to be used for backprop.
        Must return a `torch.optim.XXX` instance.
        You may choose to add certain hyperparameters of the optimizers to the `train.py` as well.
        In our experiments, we chose one good value of optimizer hyperparameters for all experiments.
        """
        return torch.optim.Adam(self.model.parameters(), lr=2e-3)
