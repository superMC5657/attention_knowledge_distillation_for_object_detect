# -*- coding: utf-8 -*-
# !@time: 2020/12/9 22 23
# !@author: superMC @email: 18758266469@163.com
# !@fileName: cbam.py

from abc import ABC

import torch
from torch import nn, matmul


class ChannelAttention(nn.Module, ABC):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avgout = self.sharedMLP(self.avg_pool(x))
        # maxout = self.sharedMLP(self.max_pool(x))
        # attn = self.sigmoid(avgout + maxout)
        attn = self.sigmoid(self.sharedMLP(self.avg_pool(x) + self.max_pool(x)))
        y = x * attn.expand_as(x)
        return y, attn


class SpatialAttention(nn.Module, ABC):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avgout, maxout], dim=1)
        attn = self.sigmoid(self.conv(attn))
        y = x * attn.expand_as(x)
        return y, attn


if __name__ == '__main__':
    x = torch.rand((1, 128, 10, 10))
    ca = ChannelAttention(128)
    y1, attn1 = ca(x)
    sa = SpatialAttention()
    y2, attn2 = sa(x)
    print(y1.size(), attn1.size())
    print(y2.size(), attn2.size())
