import numpy as np
import torch

# a = np.load('data/helix_3D_train.npy')
# print(a.shape)


def time_embed_func(t):
    """
    Returns the time embedding for a given time t.
    """
    #dimension of t = [num,1]

    N = 100
    D = 4

    sins = torch.zeros(size=(t.size(0), D))
    coss = torch.zeros(size=(t.size(0), D))
    embed = torch.zeros(size=(t.size(0), D))

    denoms = torch.arange(0, D)
    denoms = torch.where((denoms%2==0), denoms, denoms-1)/D
    denoms = 1/(torch.pow(N,denoms).reshape(1,-1))


    print((t*denoms).shape)

    sins = torch.sin(t*denoms)
    coss = torch.cos(t*denoms)

    embed[:,0::2] = sins[:,0::2]
    embed[:,1::2] = coss[:,1::2]

    return embed        




 
def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P
 
P = getPositionEncoding(seq_len=50, d=4, n=100)
print(P)


Q = (time_embed_func(torch.arange(50).reshape(-1,1)))


print((Q - torch.tensor(P))**2 <= 0.001)
