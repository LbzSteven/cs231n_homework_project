# -*- coding: utf-8 -*-
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision as tv
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def imshow(inp, title=None):
    """
    Imshow for Tensor.
    """
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def data_load(path):
    """
    功能：使用pytorch装在cifar-10数据集
    输出:
        train_data(float.tensor): (50000,3,32,32),
        train_label(int.tensor): (50000),
        test_data(float.tensor): (10000,3,32,32),
        test_label(int.tensor): (10000)

    """
    dset.CIFAR10(path, train=True, transform=None, target_transform=None, download=True)
    dset.CIFAR10(path, train=False, transform=None, target_transform=None, download=True)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ])
    train_set = tv.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
    test_set = tv.datasets.CIFAR10(root=path, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=50000, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=False, num_workers=0)

    train_data_interator = enumerate(train_loader)
    test_data_interator = enumerate(test_loader)

    train_data = next(train_data_interator)
    test_data = next(test_data_interator)

    labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_data[1][0], train_data[1][1], test_data[1][0], test_data[1][1], labels
