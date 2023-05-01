
import torch




cifar = torch.load('cifar-rotate_without_pca_l4_10C_old.pth')
data = torch.Tensor(cifar['resnet18_train_features'])


s = (torch.sum(data,(1,2,3)) == 0)

print(s.shape,s.sum())



