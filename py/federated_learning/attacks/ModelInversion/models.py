import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from .utils import set_random_seed


def construct_model(model, num_classes=10, seed=None, num_channels=3, modelkey=None):
    if modelkey is None:
        if seed is None:
            model_init_seed = np.random.randint(0, 2**32 - 10)
        else:
            model_init_seed = seed
    else:
        model_init_seed = modelkey
    set_random_seed(model_init_seed)
        
    if model == 'LeNet':
        model =LeNet_Small()
    else:
        raise NotImplementedError()
    return model, model_init_seed


class LeNet_Small(nn.Module):
    def __init__(self):
        super(LeNet_Small, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=5, stride=1, bias=False)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=False)
        self.act2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        # LeNet use 5 for 32x32. For 28x28, we adjust to 4.
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=4, stride=1, bias=False)
        self.act3 = nn.ReLU()
        self.linear1 = nn.Linear(in_features=480, out_features=84)
        self.act4 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=84, out_features=10)
    
    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.act4(x)
        x = self.linear2(x)
        x = self.dequant(x)
        return x