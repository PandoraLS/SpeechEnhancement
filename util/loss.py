# -*- coding: utf-8 -*-
# Author：lisen
# Date：20-3-8 下午2:27

import torch
import torch.nn as nn
import numpy as np

EPS = np.finfo(float).eps

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, deg, clean):
        return torch.sqrt(torch.mean(torch.pow((deg - clean), 2)) + EPS)


class SDRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, deg, clean):
        loss_sdr = -1. * torch.mean(deg * clean) ** 2 / (torch.mean(deg ** 2) + 2e-7)
        return loss_sdr


def rmse_loss():
    return RMSELoss()


def mse_loss():
    return torch.nn.MSELoss()
