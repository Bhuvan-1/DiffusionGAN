import torch
import numpy as np
from scipy.spatial.distance import cdist
from pyemd import emd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

def gaussian_kernel(x, x0, temperature=1e-1):
    dim = x0.size(1)
    x = x.view((1, -1))
    exp_term = torch.sum(- 0.5 * (x - x0) ** 2, dim=1)
    main_term = torch.exp(exp_term / (2 * temperature))
    coeff = 1. / torch.sqrt(torch.Tensor([2 * torch.pi * temperature])) ** dim
    prod = coeff * main_term
    return torch.sum(prod) / x0.size(0)

def get_likelihood(data, pred, temperature):
    lh = torch.zeros(pred.size(0))
    dim = pred.size(1)
    for i in range(pred.size(0)):
        lh[i] = gaussian_kernel(pred[i,:], data, temperature)
    return torch.mean(lh)

def get_ll(data, pred, temperature=1e-1):
    return torch.log(get_likelihood(data, pred, temperature))

def get_nll(data, pred, temperature=1e-1):
    return -get_ll(data, pred, temperature)

def get_nll_bits_per_dim(data, pred, temperature=1e-1):
    return get_nll(data, pred, temperature) / (torch.log(torch.Tensor([2])) * data.shape[0])

def get_emd(d1, d2):
    d_comb = np.concatenate((d1, d2), axis=0)
    dist = np.linalg.norm((d_comb), axis=1).reshape((-1,1))
    d1 = np.concatenate((np.zeros((d1.shape[0], 1)), d1), axis=1)
    d2 = np.concatenate((np.ones((d2.shape[0], 1)), d2), axis=1)
    d_comb = np.concatenate((d1, d2), axis=0)
    app = np.concatenate((dist, d_comb), axis=1)
    app = app[app[:, 0].argsort()]
    d1_sig, d2_sig = 1 - app[:, 1], app[:, 1]
    dist_sorted = app[:, 2:]
    dist = cdist(dist_sorted, dist_sorted)
    d1_sig = d1_sig.copy(order='C')
    d2_sig = d2_sig.copy(order='C')
    dist = dist.copy(order='C')
    return emd(d1_sig, d2_sig, dist)

def plot_final_distributions(fname, testdata, gendata):
    plt.close()
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(testdata[:, 0], testdata[:, 1], testdata[:, 2], marker='+', c=testdata[:, 2], cmap=cm.spring, alpha=0.5)
    ax.scatter(gendata[:, 0], gendata[:, 1], gendata[:, 2], marker='.', c=gendata[:, 2], cmap=cm.cool, alpha=0.1)
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_zlim([-2.5, 2.5])
    fig.savefig(fname, dpi=300, bbox_inches='tight')

def animate_intermediate_distributions(fname, testdata, intermediate, track_max=False, track_min=False, smoothed_end=True):
    plt.close()
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(testdata[:, 0], testdata[:, 1], testdata[:, 2], marker='+', c=testdata[:, 2], cmap=cm.spring, alpha=0.5)
    diffused = ax.scatter(intermediate[0][:, 0], intermediate[0][:, 1], intermediate[0][:, 2], marker='.', c=intermediate[0][:, 2], cmap=cm.cool, alpha=0.1)
    if track_max:
        max_x, max_y, max_z = [], [], []
        max_idx = torch.argmax(intermediate[-1][:, 2])
        max_trace, = ax.plot(max_x, max_y, max_z, color='blue')
    if track_min:
        min_x, min_y, min_z = [], [], []
        min_idx = torch.argmin(intermediate[-1][:, 2])
        min_trace, = ax.plot(min_x, min_y, min_z, color='red')
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_zlim([-2.5, 2.5])

    fig.set_size_inches(5, 5)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    if smoothed_end:
        # Repeat last a couple times for nicer animation
        for _ in range(len(intermediate) // 5):
            intermediate.append(intermediate[-1])

    def animate_diffused(i):
        global max_x, max_y, max_z, min_x, min_y, min_z
        # https://stackoverflow.com/a/41609238
        diffused._offsets3d = (intermediate[i][:, 0].detach().cpu().numpy(), intermediate[i][:, 1].detach().cpu().numpy(), intermediate[i][:, 2].detach().cpu().numpy())
        diffused._c = intermediate[i][:, 2].detach().cpu().numpy()

        if i == 0:
            if track_max:
                max_x, max_y, max_z = [], [], []
            if track_min:
                min_x, min_y, min_z = [], [], []
        if track_max:
            max_x.append(intermediate[i][max_idx, 0])
            max_y.append(intermediate[i][max_idx, 1])
            max_z.append(intermediate[i][max_idx, 2])
            max_trace.set_data(max_x, max_y)
            max_trace.set_3d_properties(max_z)
        if track_min:
            min_x.append(intermediate[i][min_idx, 0])
            min_y.append(intermediate[i][min_idx, 1])
            min_z.append(intermediate[i][min_idx, 2])
            min_trace.set_data(min_x, min_y)
            min_trace.set_3d_properties(min_z)

        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        fig.tight_layout()

        ret = (diffused,)
        if track_max:
            ret = ret + (max_trace,)
        if track_min:
            ret = ret + (min_trace,)
        return ret

    anim = animation.FuncAnimation(fig, animate_diffused, repeat=True, frames=len(intermediate)-1, interval=50)
    writer = animation.PillowWriter(fps=60,
                                    metadata=dict(artist='CS726-2023 diffusion model HW2'),
                                    bitrate=1800)
    def print_anim_progress(i, n):
        msg = 'Starting GIF creation' if i == n else f'Rendering frame {i}/{n}'
        print(msg, end='\r', flush=True)
    anim.save(fname, writer=writer, dpi=100, progress_callback=print_anim_progress)
    print(f'\rAnimation written to "{fname}"\n')
