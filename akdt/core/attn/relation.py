# -*- coding: utf-8 -*-
# !@time: 2020/12/16 下午2:01
# !@author: superMC @email: 18758266469@163.com
# !@fileName: relation.py
from abc import ABC

from torch import nn


class ClassAttention(nn.Module, ABC):
    def __init__(self):
        super(ClassAttention, self).__init__()
