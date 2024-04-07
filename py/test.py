# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
print(torch.__version__, torchvision.__version__)

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)
        
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 100)
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out

def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

device = "cpu"
print("Running on %s" % device)

dst = datasets.CIFAR100("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = 3
gt_data = tp(dst[img_index][0]).to(device)


gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label)

plt.imshow(tt(gt_data[0].cpu()))

net = LeNet().to(device)


torch.manual_seed(1234)

# net.apply(weights_init)
criterion = cross_entropy_for_onehot

# compute original gradient 
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())

original_dy_dx = list((_.detach().clone() for _ in dy_dx))

# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

plt.imshow(tt(dummy_data[0].cpu()))
plt.show()

# optimizer = torch.optim.LBFGS([dummy_data, dummy_label])


# history = []
# for iters in range(300):
#     def closure():
#         optimizer.zero_grad()

#         dummy_pred = net(dummy_data) 
#         dummy_onehot_label = F.softmax(dummy_label, dim=-1)
#         dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
#         dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
#         grad_diff = 0
#         for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
#             grad_diff += ((gx - gy) ** 2).sum()
#         grad_diff.backward()
        
#         return grad_diff
    
#     optimizer.step(closure)
#     if iters % 10 == 0: 
#         current_loss = closure()
#         print(iters, "%.4f" % current_loss.item())
#         history.append(tt(dummy_data[0].cpu()))

# plt.figure(figsize=(12, 8))
# for i in range(30):
#     plt.subplot(3, 10, i + 1)
#     plt.imshow(history[i])
#     plt.title("iter=%d" % (i * 10))
#     plt.axis('off')

# plt.show()