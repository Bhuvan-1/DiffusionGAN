import numpy as np
from torch.utils.data import Dataset

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ThreeDSinDataset(Dataset):
    def __init__(self, npy_path, mean=None, std=None):
        super().__init__()
        self.data = np.load(npy_path)
        if mean is None:
            mean = np.mean(self.data, axis=0)
        if std is None:
            std = np.std(self.data, axis=0)
        self.data = (self.data - mean) / std

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

class CIFAR10(Dataset):
    def __init__(self, npy_path, mean=None, std=None):
        super().__init__()

        seed_torch(0)

        self.data = np.load(npy_path)
        if mean is None:
            mean = np.mean(self.data, axis=0)
        if std is None:
            std = np.std(self.data, axis=0)

        self.mean = mean
        self.std = std
        
        self.data = (self.data - mean) / std

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]



