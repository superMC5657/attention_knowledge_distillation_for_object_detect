# -*- coding: utf-8 -*-
# !@time: 2020/12/8 下午9:30
# !@author: superMC @email: 18758266469@163.com
# !@fileName: se.py

from abc import ABC

import torch
from torch import nn


class CSEModule(nn.Module, ABC):
    def __init__(self, in_planes, hidden_ratio=16):
        hidden_state = in_planes // hidden_ratio
        super().__init__()
        self.global_average_pool = nn.AdaptiveAvgPool2d(1)
        self.in_planes = in_planes
        self.fc = nn.Sequential(
            nn.Linear(in_planes, hidden_state, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_state, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_num, height, width = x.size()
        attn = self.global_average_pool(x).view(batch_size, channel_num)
        attn = self.fc(attn).view(batch_size, channel_num, 1, 1)
        y = x * attn.expand_as(x)
        return y, attn


class SSEModule(nn.Module, ABC):
    def __init__(self, in_planes, HxW, hidden_ratio=8, heatmap=False):
        super().__init__()
        hidden_state = HxW // hidden_ratio
        self.heatmap = None
        if heatmap:
            self.heatmap = nn.Conv2d(in_planes, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(HxW, hidden_state, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_state, HxW, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_num, width, height = x.size()
        if self.heatmap:
            attn = self.heatmap(x).view(batch_size, -1)
        else:
            attn = torch.mean(x, dim=1, keepdim=True).view(batch_size, -1)
        attn = self.fc(attn).view(batch_size, 1, width, height)
        y = x * attn.expand_as(x)
        return y, attn


if __name__ == '__main__':
    x = torch.ones((1, 128, 10, 10))
    cse = CSEModule(128)
    y1, attn1 = cse(x)
    sse = SSEModule(128, 100)
    y2, attn2 = sse(x)
    print(y1.size(), attn1.size())
    print(y2.size(), attn2.size())
