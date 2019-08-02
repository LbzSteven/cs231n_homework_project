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
        train_set
        test_set
    """
    dset.CIFAR10(path, train=True, transform=None, target_transform=None, download=True)
    dset.CIFAR10(path, train=False, transform=None, target_transform=None, download=True)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ])
    train_set = tv.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
    test_set = tv.datasets.CIFAR10(root=path, train=False, download=True, transform=transform)

    return train_set, test_set

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)
