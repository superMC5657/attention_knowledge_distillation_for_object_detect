# -*- coding: utf-8 -*-
# !@time: 2020/12/16 上午12:18
# !@author: superMC @email: 18758266469@163.com
# !@fileName: demo.py


import torch
from torch import matmul

x1 = [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]
x1 = torch.Tensor(x1)
x2 = x1
x3 = matmul(x1, x2)
print(x3)
