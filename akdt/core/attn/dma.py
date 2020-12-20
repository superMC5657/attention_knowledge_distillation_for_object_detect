# -*- coding: utf-8 -*-
# !@time: 2020/12/11 下午3:36
# !@author: superMC @email: 18758266469@163.com
# !@fileName: dma.py
from abc import ABC

import torch

from torch import nn, matmul


class MultiHeadSpatialAttention(nn.Module, ABC):
    def __init__(self, in_planes, hidden_ratio=16, multi_head=4, alpha=0.5, fuse=False):
        super(MultiHeadSpatialAttention, self).__init__()
        hidden_state = in_planes // hidden_ratio
        hidden_state = multi_head * hidden_state
        self.multi_head = multi_head
        self.hidden_state = hidden_state

        self.query_conv = nn.Conv2d(in_channels=in_planes, out_channels=hidden_state, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_planes, out_channels=hidden_state, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_planes, out_channels=hidden_state, kernel_size=1)
        self.conv = nn.Conv2d(in_channels=hidden_state, out_channels=in_planes, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_planes)
        if fuse and multi_head > 1:
            self.fuse = nn.Conv2d(in_channels=multi_head, out_channels=multi_head, kernel_size=1, bias=False)
        else:
            self.fuse = None
        self.attention = ScaledDotProductAttention(temperature=hidden_state ** alpha, fuse=self.fuse)

    def forward(self, x):
        batch_size, channel_num, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, self.multi_head, -1, width * height).transpose(2,
                                                                                                        3).contiguous()
        proj_key = self.key_conv(x).view(batch_size, self.multi_head, -1, width * height)
        out = self.value_conv(x).view(batch_size, self.multi_head, -1, width * height).transpose(2,
                                                                                                 3).contiguous()

        out, attn = self.attention(proj_query, proj_key, out)
        out = out.transpose(2, 3).contiguous().view(batch_size, -1, width, height)
        out = self.conv(out)
        out = self.bn(out)

        return out, attn


class MultiHeadChannelAttention(nn.Module, ABC):
    def __init__(self, in_planes, hidden_ratio=8, multi_head=4, beta=0.5, fuse=False):
        super(MultiHeadChannelAttention, self).__init__()
        hidden_state = in_planes // hidden_ratio
        hidden_state = multi_head * hidden_state
        self.multi_head = multi_head
        self.hidden_state = hidden_state

        self.query_conv = nn.Conv2d(in_channels=in_planes, out_channels=hidden_state, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_planes, out_channels=hidden_state, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_planes, out_channels=hidden_state, kernel_size=1)

        self.conv = nn.Conv2d(in_channels=hidden_state, out_channels=in_planes, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_planes)
        if fuse and multi_head > 1:
            self.fuse = nn.Conv2d(in_channels=multi_head, out_channels=multi_head, kernel_size=1)
        else:
            self.fuse = None

        self.attention = ScaledDotProductAttention(temperature=hidden_state ** beta, fuse=self.fuse)

    def forward(self, x):
        batch_size, channel_num, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, self.multi_head, -1, width * height)
        proj_key = self.key_conv(x).view(batch_size, self.multi_head, -1, width * height).transpose(2, 3).contiguous()
        proj_value = self.value_conv(x).view(batch_size, self.multi_head, -1, width * height)

        out, attn = self.attention(proj_query, proj_key, proj_value)
        out = out.view(batch_size, -1, width, height)
        out = self.conv(out)
        out = self.bn(out)

        return out, attn


class ScaledDotProductAttention(nn.Module, ABC):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1, fuse=None):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout, inplace=False)
        self.softmax = nn.Softmax(dim=-1)
        self.fuse = fuse

    # def forward(self, q, k, v):
    #     batch_size, multi_head, d_v, WxH = v.size()
    #     attn = torch.bmm(q.view(-1, q.size(2), q.size(3)) / self.temperature, k.view(-1, k.size(2), k.size(3)))
    #     if self.fuse:
    #         attn = attn.view(batch_size, multi_head, attn.size(1), attn.size(2))
    #         attn = self.fuse(attn)
    #         attn = attn.view(-1, attn.size(2), attn.size(3))
    #     attn = self.dropout(self.softmax(attn))
    #     # dropout 前softmax的话 inplace必须为false
    #     v = torch.bmm(attn, v.view(-1, d_v, WxH))
    #
    #     return v.view(batch_size, multi_head, d_v, WxH), attn.view(batch_size, multi_head, attn.size(1), attn.size(2))

    def forward(self, q, k, v):
        attn = matmul(q / self.temperature, k)
        if self.fuse:
            attn = self.fuse(attn)
        attn = self.dropout(self.softmax(attn))
        v = matmul(attn, v)
        return v, attn


if __name__ == '__main__':
    from torchsummaryX import summary
    from thop import profile
    from fvcore.nn import flop_count

    x = torch.rand((1, 32, 20, 20))
    mca = MultiHeadChannelAttention(32, 4, 4, fuse=False)
    msa = MultiHeadSpatialAttention(32, 4, 4, fuse=False)
    # flops, skip = flop_count(msa, (x,))
    # print("%s|%s" % (flops, skip))
    # out, attn = msa(x)
    # print(out.size(), attn.size())

    # summary(mca, (32, 20, 20), device='cpu')
    # summary(msa, (32, 20, 20), device='cpu')

    # summary(mca, x)
    # summary(msa, x)
    flops, params = profile(mca, inputs=(x,))
    print(flops, params)
    flops, params = profile(msa, inputs=(x,))
    print(flops, params)
