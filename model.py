import torch
import pytorch_lightning as pl

class LitDiffusionModel(pl.LightningModule):
    def __init__(self, n_dim=3, n_steps=200, lbeta=1e-5, ubeta=1e-2, n_layers=5,embedD=4,embedN = 100,hidden_size = 256,noise='linear'):
        super().__init__()
        """
        If you include more hyperparams (e.g. `n_layers`), be sure to add that to `argparse` from `train.py`.
        Also, manually make sure that this new hyperparameter is being saved in `hparams.yaml`.
        """
        self.save_hyperparameters()

        """
        Your model implementation starts here. We have separate learnable modules for `time_embed` and `model`.
        You may choose a different architecture altogether. Feel free to explore what works best for you.
        If your architecture is just a sequence of `torch.nn.XXX` layers, using `torch.nn.Sequential` will be easier.
        
        `time_embed` can be learned or a fixed function based on the insights you get from visualizing the data.
        If your `model` is different for different datasets, you can use a hyperparameter to switch between them.
        Make sure that your hyperparameter behaves as expecte and is being saved correctly in `hparams.yaml`.
        """

        # self.time_embed = lambda t : torch.tensor([ torch.sin(t/self.embedN**(i/self.embedD)) if i%2 == 0 else torch.cos(t/self.embedN**( (i-1)/self.embedD )) for i in range(self.embedD)  ])
        # self.time_embed = torch.nn.Sequential(
        #     torch.nn.Linear(1, embedD),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(embedD, embedD),
        # )
        self.time_embed = self.time_embed_func

        layers = [torch.nn.Linear(n_dim + embedD, hidden_size), torch.nn.ReLU()]
        for i in range(n_layers-2):
            layers.extend([torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()])
        layers.append(torch.nn.Linear(hidden_size, n_dim))
        self.model = torch.nn.Sequential(*layers)

        """
        Be sure to save at least these 2 parameters in the model instance.
        """
        self.lbeta = lbeta
        self.ubeta = ubeta

        self.n_dim = n_dim
        self.n_steps = n_steps

        self.n_layers = n_layers
        self.hidden_size  = hidden_size

        self.embedD = embedD
        self.embedN = embedN

        self.noise = noise

        """
        Sets up variables for noise schedule
        """
        self.init_alpha_beta_schedule(lbeta, ubeta)

    def forward(self, x, t):
        """
        Similar to `forward` function in `nn.Module`. 
        Notice here that `x` and `t` are passed separately. If you are using an architecture that combines
        `x` and `t` in a different way, modify this function appropriately.
        """

        if(len(x.shape) == 1):
            x = x.reshape(1,-1)
        
        if not isinstance(t, torch.Tensor):
            t = torch.LongTensor([t]).expand(x.size(0))

        t = t.reshape(x.size(0),-1).to(torch.float64)
        t_embed = self.time_embed(t)

        return self.model(torch.cat((x, t_embed), dim=1).float())

    def init_alpha_beta_schedule(self, lbeta, ubeta):
        """
        Set up your noise schedule. You can perhaps have an additional hyperparameter that allows you to
        switch between various schedules for answering q4 in depth. Make sure that this hyperparameter 
        is included correctly while saving and loading your checkpoints.
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

        


    def q_sample(self, x, t):
        """
        Sample from q given x_t.
        """
        pass

    def p_sample(self, x, t):
        """
        Sample from p given x_t.
        """
        pass

    def training_step(self, batch, batch_idx):
        """
        Implements one training step.
        Given a batch of samples (n_samples, n_dim) from the distribution you must calculate the loss
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

        #-----BATCH SIZE :[N,n_dim]
        epsilons = torch.randn(batch.size(0), self.n_dim)
        times = torch.randint(1, self.n_steps+1, (batch.size(0),1))

        time_embeds = self.time_embed(times.to(torch.float64))

        reparams = torch.sqrt(self.alpha_bar[times-1])*batch + torch.sqrt(1 - self.alpha_bar[times-1])*epsilons
        eps_thetas = self.forward(reparams, time_embeds)
        loss = torch.sum((epsilons - eps_thetas)**2)

        return loss

    def sample(self, n_samples, progress=False, return_intermediate=False):
        """
        Implements inference step for the DDPM.
        `progress` is an optional flag to implement -- it should just show the current step in diffusion
        reverse process.
        If `return_intermediate` is `False`,
            the function returns a `n_samples` sampled from the learned DDPM
            i.e. a Tensor of size (n_samples, n_dim).
            Return: (n_samples, n_dim)(final result from diffusion)
        Else
            the function returns all the intermediate steps in the diffusion process as well 
            i.e. a Tensor of size (n_samples, n_dim) and a list of `self.n_steps` Tensors of size (n_samples, n_dim) each.
            Return: (n_samples, n_dim)(final result), [(n_samples, n_dim)(intermediate) x n_steps]
        """

        X = torch.zeros(n_samples,self.n_steps+1,self.n_dim)
        X[:,-1,:] = torch.randn(n_samples, self.n_dim)

        self.sigma = torch.sqrt(self.beta)
        sigma = self.sigma
        alpha_bar = self.alpha_bar
        alpha = self.alpha

        for t in range(self.n_steps, 0, -1):

            times = torch.LongTensor([t]).expand(n_samples).reshape(n_samples,1).to(torch.float64)

            X[:,t-1,:] =  (1/torch.sqrt(alpha[t-1]))*( X[:,t,:] - self.forward(X[:,t,:], self.time_embed(times) ) * ( (1 - alpha[t-1]) / torch.sqrt(1 - alpha_bar[t-1]) ) )
            if(t > 1):
                X[:,t-1,:] = X[:,t-1,:] + torch.randn(n_samples, self.n_dim) * sigma[t-1]
        
        if(not return_intermediate):
            return X[:,0,:]
        else:
            return ( X[:,0,:], [ X[:,i,:] for i in range(self.n_steps,0,-1) ] )

    def configure_optimizers(self):
        """
        Sets up the optimizer to be used for backprop.
        Must return a `torch.optim.XXX` instance.
        You may choose to add certain hyperparameters of the optimizers to the `train.py` as well.
        In our experiments, we chose one good value of optimizer hyperparameters for all experiments.
        """
        return torch.optim.Adam(self.model.parameters(), lr=2e-3)

    def time_embed_func(self, t):
        """
        Returns the time embedding for a given time t.
        """
        #dimension of t = [num,1]

        N = self.embedN
        D = self.embedD

        sins = torch.zeros(size=(t.size(0), D))
        coss = torch.zeros(size=(t.size(0), D))
        embed = torch.zeros(size=(t.size(0), D))

        denoms = torch.arange(0, D)
        denoms = torch.where((denoms%2==0), denoms, denoms-1)/D
        denoms = (N**denoms).reshape(1,-1)
        denoms * 1/denoms

        sins = torch.sin(t*denoms)
        coss = torch.cos(t*denoms)

        embed[:,0::2] = sins[:,0::2]
        embed[:,1::2] = coss[:,1::2]

        return embed        


