#!/usr/bin/env python
from models import register
import os
import sys
import tqdm
import pdb

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F


class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity

        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0

        return torch.cos(omega) * torch.exp(-(scale ** 2))


class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity

        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=True):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)

    def forward(self, input):
        # print("input.shape",input.shape)
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin

        return torch.exp(1j * omega - scale.abs().square())

@register('wire')
class wire(nn.Module):
    def __init__(self, in_dim, hidden_features,
                 hidden_layers,
                 out_features, outermost_linear=True,
                 first_omega_0=10, hidden_omega_0=10., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()

        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer

        # Since complex numbers are two real numbers, reduce the number of 
        # hidden parameters by 2
        hidden_features = int(hidden_features / np.sqrt(2))
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = 'gabor'

        # Legacy parameter
        self.pos_encode = False
        self.head = [] # 在输入坐标后 MLP的前面加一个编码
        self.body = []
        # self.tail = []
        # print("in_dim",in_dim)
        self.head.append(self.nonlin(in_dim,
                                    hidden_features,
                                    omega0=first_omega_0,
                                    sigma0=scale,
                                    is_first=True,
                                    trainable=False))

        for i in range(hidden_layers):
            self.body.append(self.nonlin(hidden_features,
                                        hidden_features,
                                        omega0=hidden_omega_0,
                                        sigma0=scale))

        self.tail = nn.Linear(hidden_features,
                                 out_features,
                                 dtype=dtype)
        # self.net.append(final_linear)
        self.head = nn.Sequential(*self.head)
        self.body = nn.Sequential(*self.body)


    def forward(self, coords):
        # print(coords,coords.shape)
        # print("coords.shape=",coords.shape)
        output = self.head(coords)
        # print("111",output.shape)
        output = self.body(output)
        output = self.tail(output)

        if self.wavelet == 'gabor':
            return output.real

        return output