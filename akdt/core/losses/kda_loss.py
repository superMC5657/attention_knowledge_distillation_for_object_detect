# -*- coding: utf-8 -*-
# !@time: 2020/12/16 21 41
# !@author: superMC @email: 18758266469@163.com
# !@fileName: kda_loss.py
from abc import ABC

import torch
from torch import nn


class SpatialAttentionLoss(nn.Module, ABC):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, f1, f2):
        pass


class ChannelAttentionLoss(nn.Module, ABC):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def forward(self, f1, f2):
        batch_size, multi_head, channel_num, channel_num = f1.size()
        f1 = f1.view(-1, channel_num)
        f2 = f2.view(-1, channel_num)

        return loss * self.beta


class RelationFeatureLoss(nn.Module, ABC):
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, f1, f2):
        pass


if __name__ == '__main__':
    f1 = torch.ones((1, 2, 4, 4))
    f2 = torch.ones((1, 2, 4, 4))
    cal = ChannelAttentionLoss()
    loss = cal(f1, f2)
    print(loss)
