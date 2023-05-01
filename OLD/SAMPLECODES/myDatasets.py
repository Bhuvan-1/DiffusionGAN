import os
import PIL
import torch
import pickle
import random
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class CIFAR10(Dataset):
    def __init__(self, train):
        super(CIFAR10, self).__init__()
        if train:
            seed_torch(0)
        else:
            seed_torch(1)
        self.train = train
        self.num_attributes = 128 #128,8,8
        self.num_features = 128 #128,8,8
        self.num_classes = 10
        if self.train:
            self.data_size = 50000
            # cifar = torch.load('../18_cifar/PyTorch_CIFAR10/cifar_without_pca_l4.pth')
            cifar = torch.load('./PyTorch_CIFAR10/cifar_without_pca_l4.pth')
            # self.data = torch.Tensor(cifar['final_train_features'])
            self.data = torch.Tensor(cifar['resnet18_train_features'])
            self.labels = torch.Tensor(cifar['train_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        else:
            self.data_size = 10000
            # cifar = torch.load('../18_cifar/PyTorch_CIFAR10/cifar_without_pca_l4.pth')
            cifar = torch.load('./PyTorch_CIFAR10/cifar_without_pca_l4.pth')
            # self.data = torch.Tensor(cifar['final_test_features'])
            self.data = torch.Tensor(cifar['resnet18_test_features'])
            self.labels = torch.Tensor(cifar['test_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        
    def __getitem__(self, i):
        return i, self.data[i], self.labels[i]
        
    def __len__(self):
        return self.data.shape[0]

    # def nearest(self,x):
    #     idx = torch.searchsorted(self.sortedX, x)
    #     return self.indices[idx]

    def plot(self):
        pass
        # plt.show()
        # plt.savefig('data3.png')

class CIFAR10_ROTATE(Dataset):
    def __init__(self, train):
        super(CIFAR10_ROTATE, self).__init__()
        if train:
            seed_torch(0)
        else:
            seed_torch(1)
        self.train = train
        self.num_attributes = 128 #128,8,8
        self.num_features = 128 #128,8,8
        self.num_classes = 2
        if self.train:
            self.data_size = 25000
            # cifar = torch.load('../18_cifar/PyTorch_CIFAR10/cifar-rotate_without_pca_l4_half.pth')
            cifar = torch.load('./PyTorch_CIFAR10-ROTATE/cifar-rotate_without_pca_l4_10C_half_2.pth')
            # self.data = torch.Tensor(cifar['final_train_features'])
            self.data = torch.Tensor(cifar['resnet18_train_features'])
            self.labels = torch.Tensor(cifar['train_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        else:
            self.data_size = 10000
            # cifar = torch.load('../18_cifar/PyTorch_CIFAR10/cifar-rotate_without_pca_l4_half.pth')
            cifar = torch.load('./PyTorch_CIFAR10-ROTATE/cifar-rotate_without_pca_l4_10C_half_2.pth')
            # self.data = torch.Tensor(cifar['final_test_features'])
            self.data = torch.Tensor(cifar['resnet18_test_features'])
            self.labels = torch.Tensor(cifar['test_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        
    def __getitem__(self, i):
        return i, self.data[i], self.labels[i]
        
    def __len__(self):
        return self.data.shape[0]

    # def nearest(self,x):
    #     idx = torch.searchsorted(self.sortedX, x)
    #     return self.indices[idx]

    def plot(self):
        pass
        # plt.show()
        # plt.savefig('data3.png')

class TINY_IMAGENET_ROTATE(Dataset):
    def __init__(self, train):
        super(TINY_IMAGENET_ROTATE, self).__init__()
        if train:
            seed_torch(0)
        else:
            seed_torch(1)
        self.train = train
        self.num_attributes = 128 #128,8,8
        self.num_features = 128 #128,8,8
        self.num_classes = 2
        if self.train:
            self.data_size = 25000
            cifar = torch.load('./PyTorch_TINY_IMAGENET-ROTATE/tiny-imagenet_l4_25_10K_blur.pth')
            self.data = torch.Tensor(cifar['resnet18_train_features'])
            self.labels = torch.Tensor(cifar['train_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        else:
            self.data_size = 10000
            cifar = torch.load('./PyTorch_TINY_IMAGENET-ROTATE/tiny-imagenet_l4_25_10K_blur.pth')            
            self.data = torch.Tensor(cifar['resnet18_test_features'])
            self.labels = torch.Tensor(cifar['test_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        
    def __getitem__(self, i):
        return i, self.data[i], self.labels[i]
        
    def __len__(self):
        return self.data.shape[0]

    # def nearest(self,x):
    #     idx = torch.searchsorted(self.sortedX, x)
    #     return self.indices[idx]

    def plot(self):
        pass
        # plt.show()
        # plt.savefig('data3.png')

class STL10_ROTATE(Dataset):
    def __init__(self, train):
        super(STL10_ROTATE, self).__init__()
        if train:
            seed_torch(0)
        else:
            seed_torch(1)
        self.train = train
        self.num_attributes = 128 #128,8,8
        self.num_features = 128 #128,8,8
        self.num_classes = 2
        if self.train:
            self.data_size = 16000*2
            cifar = torch.load('./PyTorch_STL10-ROTATE/stl_l4_32_10K_half.pth')
            self.data = torch.Tensor(cifar['resnet18_train_features'])
            self.labels = torch.Tensor(cifar['train_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        else:
            self.data_size = 10000
            cifar = torch.load('./PyTorch_STL10-ROTATE/stl_l4_32_10K_half.pth')            
            self.data = torch.Tensor(cifar['resnet18_test_features'])
            self.labels = torch.Tensor(cifar['test_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        
    def __getitem__(self, i):
        return i, self.data[i], self.labels[i]
        
    def __len__(self):
        return self.data.shape[0]

    # def nearest(self,x):
    #     idx = torch.searchsorted(self.sortedX, x)
    #     return self.indices[idx]

    def plot(self):
        pass
        # plt.show()
        # plt.savefig('data3.png')


class CIFAR100_ROTATE(Dataset):
    def __init__(self, train):
        super(CIFAR100_ROTATE, self).__init__()
        if train:
            seed_torch(0)
        else:
            seed_torch(1)
        self.train = train
        self.num_attributes = 128 #128,8,8
        self.num_features = 128 #128,8,8
        self.num_classes = 2
        if self.train:
            self.data_size = 50000
            # cifar = torch.load('../18_cifar/PyTorch_CIFAR10/cifar-rotate_without_pca_l4_half.pth')
            cifar = torch.load('./PyTorch_CIFAR100-ROTATE/cifar100-rotate_without_pca_l4.pth')
            # self.data = torch.Tensor(cifar['final_train_features'])
            self.data = torch.Tensor(cifar['resnet18_train_features'])
            self.labels = torch.Tensor(cifar['train_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        else:
            self.data_size = 10000
            # cifar = torch.load('../18_cifar/PyTorch_CIFAR10/cifar-rotate_without_pca_l4_half.pth')
            cifar = torch.load('./PyTorch_CIFAR100-ROTATE/cifar100-rotate_without_pca_l4.pth')
            # self.data = torch.Tensor(cifar['final_test_features'])
            self.data = torch.Tensor(cifar['resnet18_test_features'])
            self.labels = torch.Tensor(cifar['test_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        
    def __getitem__(self, i):
        return i, self.data[i], self.labels[i]
        
    def __len__(self):
        return self.data.shape[0]

    # def nearest(self,x):
    #     idx = torch.searchsorted(self.sortedX, x)
    #     return self.indices[idx]

    def plot(self):
        pass
        # plt.show()
        # plt.savefig('data3.png')

class PathMNIST(Dataset):
    def __init__(self, train):
        super(PathMNIST, self).__init__()
        if train:
            seed_torch(0)
        else:
            seed_torch(1)
        self.train = train
        self.num_attributes = 64 # 64,7,7
        self.num_features = 64 # 64,7,7
        self.num_classes = 9
        if self.train:
            self.data_size = 89996
            # cifar = torch.load('../21_pathmnist/PyTorch_PathMNIST/pathmnist_without_pca_l5.pth')
            cifar = torch.load('./PyTorch_PathMNIST/pathmnist_without_pca_l5.pth')
            # self.data = torch.Tensor(cifar['final_train_features'])
            self.data = torch.Tensor(cifar['resnet18_train_features'])
            self.labels = torch.Tensor(cifar['train_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        else:
            self.data_size = 7180
            # cifar = torch.load('../21_pathmnist/PyTorch_PathMNIST/pathmnist_without_pca_l5.pth')
            cifar = torch.load('./PyTorch_PathMNIST/pathmnist_without_pca_l5.pth')
            # self.data = torch.Tensor(cifar['final_test_features'])
            self.data = torch.Tensor(cifar['resnet18_test_features'])
            self.labels = torch.Tensor(cifar['test_labels'])
            self.labels = self.labels.type(torch.LongTensor)

            idx = random.sample(range(0, self.data_size), 5)
            self.test_sample_data = self.data[idx]
            self.test_sample_labels = self.labels[idx]
        
    def __getitem__(self, i):
        return i, self.data[i], self.labels[i]
        
    def __len__(self):
        return self.data.shape[0]

class DermaMNIST(Dataset):
    def __init__(self, train):
        super(DermaMNIST, self).__init__()
        if train:
            seed_torch(0)
        else:
            seed_torch(1)
        self.train = train
        self.num_attributes = 64 # 64,7,7
        self.num_features = 64 # 64,7,7
        self.num_classes = 7
        if self.train:
            self.data_size = 7007
            # cifar = torch.load('../22_dermamnist/PyTorch_DermaMNIST/dermamnist_without_pca_l5.pth')
            cifar = torch.load('./PyTorch_DermaMNIST/dermamnist_without_pca_l5.pth')
            # self.data = torch.Tensor(cifar['final_train_features'])
            self.data = torch.Tensor(cifar['resnet18_train_features'])
            self.labels = torch.Tensor(cifar['train_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        else:
            self.data_size = 2005
            # cifar = torch.load('../22_dermamnist/PyTorch_DermaMNIST/dermamnist_without_pca_l5.pth')
            cifar = torch.load('./PyTorch_DermaMNIST/dermamnist_without_pca_l5.pth')
            # self.data = torch.Tensor(cifar['final_test_features'])
            self.data = torch.Tensor(cifar['resnet18_test_features'])
            self.labels = torch.Tensor(cifar['test_labels'])
            self.labels = self.labels.type(torch.LongTensor)

            idx = random.sample(range(0, self.data_size), 5)
            self.test_sample_data = self.data[idx]
            self.test_sample_labels = self.labels[idx]
        
    def __getitem__(self, i):
        return i, self.data[i], self.labels[i]
        
    def __len__(self):
        return self.data.shape[0]


class FashionMNIST_ROTATE(Dataset):
    def __init__(self, train):
        super(FashionMNIST_ROTATE, self).__init__()
        if train:
            seed_torch(0)
        else:
            seed_torch(1)
        self.train = train
        self.num_attributes = 64 # 64,7,7
        self.num_features = 64 # 64,7,7
        self.num_classes = 2
        if self.train:
            self.data_size = 50000
            cifar = torch.load('./PyTorch_FashionMNIST-ROTATE/fashionmnist-rotate_without_pca_l5_50K.pth')
            # self.data = torch.Tensor(cifar['final_train_features'])
            self.data = torch.Tensor(cifar['resnet18_train_features'])
            self.labels = torch.Tensor(cifar['train_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        else:
            self.data_size = 10000
            cifar = torch.load('./PyTorch_FashionMNIST-ROTATE/fashionmnist-rotate_without_pca_l5_50K.pth')
            # self.data = torch.Tensor(cifar['final_test_features'])
            self.data = torch.Tensor(cifar['resnet18_test_features'])
            self.labels = torch.Tensor(cifar['test_labels'])
            self.labels = self.labels.type(torch.LongTensor)

            idx = random.sample(range(0, self.data_size), 5)
            self.test_sample_data = self.data[idx]
            self.test_sample_labels = self.labels[idx]
        
    def __getitem__(self, i):
        return i, self.data[i], self.labels[i]
        
    def __len__(self):
        return self.data.shape[0]



class SVHN(Dataset):
    def __init__(self, train):
        super(SVHN, self).__init__()
        if train:
            seed_torch(0)
        else:
            seed_torch(1)
        self.train = train
        self.num_attributes = 64 # 64,8,8
        self.num_features = 64 # 64,8,8
        self.num_classes = 10
        if self.train:
            self.data_size = 73257
            # cifar = torch.load('../24_svhn/PyTorch_SVHN/svhn_without_pca_l5.pth')
            cifar = torch.load('./PyTorch_SVHN/svhn_without_pca_l5.pth')
            # self.data = torch.Tensor(cifar['final_train_features'])
            self.data = torch.Tensor(cifar['resnet18_train_features'])
            self.labels = torch.Tensor(cifar['train_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        else:
            self.data_size = 26032
            # cifar = torch.load('../24_svhn/PyTorch_SVHN/svhn_without_pca_l5.pth')
            cifar = torch.load('./PyTorch_SVHN/svhn_without_pca_l5.pth')
            # self.data = torch.Tensor(cifar['final_test_features'])
            self.data = torch.Tensor(cifar['resnet18_test_features'])
            self.labels = torch.Tensor(cifar['test_labels'])
            self.labels = self.labels.type(torch.LongTensor)

            idx = random.sample(range(0, self.data_size), 5)
            self.test_sample_data = self.data[idx]
            self.test_sample_labels = self.labels[idx]
        
    def __getitem__(self, i):
        return i, self.data[i], self.labels[i]
        
    def __len__(self):
        return self.data.shape[0]

    def class_wise(self):
        count = [0]*10
        for i in range(self.data_size):
            count[self.labels[i]] = count[self.labels[i]] + 1
        print(count)


class SVHN_ROTATE(Dataset):
    def __init__(self, train):
        super(SVHN_ROTATE, self).__init__()
        if train:
            seed_torch(0)
        else:
            seed_torch(1)
        self.train = train
        self.num_attributes = 64 # 64,8,8
        self.num_features = 64 # 64,8,8
        self.num_classes = 2
        if self.train:
            self.data_size = 80000
            cifar = torch.load('./PyTorch_SVHN-ROTATE/svhn-rotate_without_pca_l5.pth')
            # self.data = torch.Tensor(cifar['final_train_features'])
            self.data = torch.Tensor(cifar['resnet18_train_features'])
            self.labels = torch.Tensor(cifar['train_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        else:
            self.data_size = 30000
            cifar = torch.load('./PyTorch_SVHN-ROTATE/svhn-rotate_without_pca_l5.pth')
            # self.data = torch.Tensor(cifar['final_test_features'])
            self.data = torch.Tensor(cifar['resnet18_test_features'])
            self.labels = torch.Tensor(cifar['test_labels'])
            self.labels = self.labels.type(torch.LongTensor)

            idx = random.sample(range(0, self.data_size), 5)
            self.test_sample_data = self.data[idx]
            self.test_sample_labels = self.labels[idx]
        
    def __getitem__(self, i):
        return i, self.data[i], self.labels[i]
        
    def __len__(self):
        return self.data.shape[0]

    def class_wise(self):
        count = [0]*10
        for i in range(self.data_size):
            count[self.labels[i]] = count[self.labels[i]] + 1
        print(count)
