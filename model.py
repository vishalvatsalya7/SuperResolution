import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

#EDSR : https://arxiv.org/pdf/1707.02921.pdf architecture used.

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

def conv(ni, nf, kernel = 3,act = False):
  layers = [nn.Conv2d(ni,nf,kernel,padding=kernel//2)]
  if act:
    layers += [nn.ReLU(True)]
  return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, layers, res_scale=1.0):
        super(ResidualBlock, self).__init__()
        self.m = nn.Sequential(*layers)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.m(x) * self.res_scale

def res_block(ni):
    return ResidualBlock([conv(ni, ni, act=True), conv(ni, ni)], 0.1)

def icnr(x, scale=2, init=nn.init.kaiming_normal):
    new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = init(subkernel)
    subkernel = subkernel.transpose(0, 1)
    subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                            subkernel.shape[1], -1)
    kernel = subkernel.repeat(1, 1, scale ** 2)
    transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)
    kernel = kernel.transpose(0, 1)
    return kernel

def upsample(ni,nf,scale):
  layers = []
  if (scale & (scale - 1)) == 0:
      for i in range(int(math.log(scale,2))):
        layers += [conv(ni, nf*4), nn.PixelShuffle(2)]
  elif scale == 3:
      layers.append(conv(ni, 9 * nf, 3))
      layers.append(nn.PixelShuffle(3))
  return nn.Sequential(*layers)

class Model(nn.Module):

    def __init__(self, ni, nf, n_resblocks, scale=1.0):
        super(Model, self).__init__()
        head_layers = [conv(ni,nf)]
        body_layers = [res_block(nf) for _ in range(n_resblocks)]
        body_layers.append(conv(nf, nf))
        tail_layers = [upsample(nf, nf, scale), nn.BatchNorm2d(nf), conv(nf,ni)]
        self.head = nn.Sequential(*head_layers)
        self.body = nn.Sequential(*body_layers)
        self.tail = nn.Sequential(*tail_layers)
        self.sub_mean = MeanShift(255)
        self.add_mean = MeanShift(255, sign=1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x
