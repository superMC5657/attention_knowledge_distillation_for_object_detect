# -*- coding: utf-8 -*-
# !@time: 2020/12/10 上午12:04
# !@author: superMC @email: 18758266469@163.com
# !@fileName: da.py
# dual attention


from abc import ABC

import torch
from torch import nn, matmul


class ChannelAttention(nn.Module, ABC):
    def __init__(self, alpha=0.1):
        super(ChannelAttention, self).__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        batch_size, channel_num, width, height = x.size()
        x1 = x.view(batch_size, channel_num, -1)
        x2 = x1.transpose(1, 2).contiguous()
        attn = torch.bmm(x1, x2)
        attn = self.softmax(attn)
        y = matmul(attn, x1).view(batch_size, channel_num, width, height)
        return y, attn


class SpatialAttention(nn.Module, ABC):
    def __init__(self, beta=0.1):
        super(SpatialAttention, self).__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        batch_size, channel_num, width, height = x.size()
        x1 = x.view(batch_size, channel_num, -1)
        x2 = x1.transpose(1, 2).contiguous()
        attn = torch.bmm(x2, x1)
        attn = self.softmax(attn)
        y = matmul(attn, x2).transpose(1, 2).contiguous().view(batch_size, channel_num, width, height)
        return y, attn


if __name__ == '__main__':
    x = torch.ones((1, 128, 10, 10))
    ca = ChannelAttention(0.1)
    y1, attn1 = ca(x)
    sa = SpatialAttention(0.1)
    y2, attn2 = sa(x)
    print(y1.size(), attn1.size())
    print(y2.size(), attn2.size())
