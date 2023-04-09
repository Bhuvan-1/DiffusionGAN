# explore-diffusion
[CS726-2023] Programming assignment exploring diffusion models

# Getting started

Steps to get started with the code:

1. Install Anaconda on your system, download from -- [`https://www.anaconda.com`](https://www.anaconda.com).
3. Clone the github repo -- `git clone https://github.com/ashutoshbsathe/explore-diffusion.git`, into some convenient folder of your choice.
4. `cd explore-diffusion`.
5. Run the command -- `conda env create --file environment.yaml`. This will setup all the required dependencies.
6. Activate the environment using `source activate cs726-env` or `conda activate cs726-env`. You are done with the setup.

# Training your model

Once you code up your model in the [`model.py`](model.py) file, you can use the provided trainer in the [`train.py`](train.py) file to train your model, as -- `python train.py`. 

You can use various command line arguments to tweak the number of epochs, batch size, etc. Please check the `train.py` file for details. You can get the full list of available hyperparameters by doing `python train.py -h` 

After completion of training you can find the checkpoint and hyperparams under the `runs` directory. A demo directory structure is shown as follows:

<img width="605" alt="image" src="https://user-images.githubusercontent.com/25797790/217562148-8f6e6b39-b8df-42b9-a338-89a471228a4e.png">

Of interest are the `last.ckpt` and `hparams.yaml` files, which will be used while evaluating the trained model.

# Evaluating your trained model

Once the trained model is available, you can use the `eval.py` file to generate the metrics and visualizations. Refer the command line arguments to 
understand further. A demo run is as follows:

```
 python eval.py --ckpt_path runs/n_dim=3,n_steps=50,lbeta=1.000e-05,ubeta=1.280e-02,batch_size=1024,n_epochs=500/last.ckpt \
                --hparams_path runs/n_dim=3,n_steps=50,lbeta=1.000e-05,ubeta=1.280e-02,batch_size=1024,n_epochs=500/lightning_logs/version_0/hparams.yaml \
                --eval_nll --vis_diffusion --vis_overlay
```

This evaluates the trained model on samples generated from $3$ runs, using only the negative log likelihood (`--eval_nll`). It also generates neat visualization of the diffusion process as `gif` animations.

<details>
 <summary>Example plot generated with <code>--vis_overlay</code>. </summary>

![image](https://user-images.githubusercontent.com/22210756/217594151-79a30d7c-f733-45e6-9b48-55e7d2479249.png)


Here, yellow-magenta points represent the original distribution and the blue-purple points indicate samples generated from a trained DDPM

</details>

<details>
 <summary>Example animation produced with <code>--vis_diffusion</code>. </summary>

![00 diffusionvis track_max=False track_min=False smoothed_end=True](https://user-images.githubusercontent.com/22210756/217595408-07e149f0-a145-4fec-8900-c5eed0f6a4c3.gif)


Here, yellow-magenta points represent the original distribution and the blue-purple points indicate samples generated from a trained DDPM. Notice how the blue-purple points slowly become closer and closer to the original distribution as the reverse process progresses.

</details>

# Acknowledgements

Special thanks to [Kanad Pardeshi](https://github.com/KanPard005) for generating the `3d_sin_5_5` and `helix` distributions and helping with the implementation of several evaluation metrics
