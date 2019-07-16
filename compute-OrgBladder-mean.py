# -*- coding:utf-8 -*-
import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Dataset import ShoeprintDataset
import torchvision.models as models

import os
import sys
import math

import numpy as np
# 这个是均值，方差计算文件
data_root = '/home/dataset/ShoeprintDataset/'
save_root = './save/'
num_workers = 1
cancer = 2

batch_size_train = 1
batch_size_test = 1
max_epoch = 800
# criterion = nn.MSELoss()

train_data = ShoeprintDataset(root=data_root, fold=0, pattern=True, formats=True, types=False)
val_data = ShoeprintDataset(root=data_root, fold=0, pattern=False, formats=True, types=False)
train_dataloader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
print('train dataset len: {}'.format(len(train_dataloader.dataset)))
val_dataloader = DataLoader(val_data, batch_size=batch_size_test, shuffle=True, num_workers=num_workers)
print('val dataset len: {}'.format(len(val_dataloader.dataset)))

Mean = np.array([0., 0., 0.]).astype(np.float32)
Std = np.array([0., 0., 0.]).astype(np.float32)
#
number = 0
for batch_datas,batch_labels in train_dataloader:
    # print(batch_datas.size(),batch_datas.type())
    # print(batch_labels.size(), batch_labels.type())
    batch_datas_np = batch_datas.numpy()
    batch_datas_np = batch_datas_np.astype(np.float32)
    means = []
    stdevs = []
    for i in range(3):
        pixels = batch_datas_np[:, i, :, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    Mean += means
    Std += stdevs
    print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))
    number += 1
    # print("means: {}".format(means))
    # print("stdevs: {}".format(stdevs))
print('transforms.Normalize(Mean = {}, Std = {})'.format(Mean/len(train_dataloader.dataset), Std/len(train_dataloader.dataset)))
print(number)

    # label = batch_labels
    # label = label.float()
    # print(batch_labels)
    # print(batch_datas)
    # print(data.size)
    # break