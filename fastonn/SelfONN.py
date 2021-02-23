import torch
import math
from torch import nn
import numpy as np
from pathlib import Path
from math import factorial
import matplotlib.pyplot as plt


class SelfONNLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,q=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.q = q
        
        self.weights = nn.Parameter(torch.Tensor(out_channels,q*in_channels,kernel_size,kernel_size)) # Q x C x K x D
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters_like_torch()
        
                
    def reset_parameters(self):
        bound = 0.01
        nn.init.uniform_(self.bias,-bound,bound)
        for q in range(self.q): nn.init.xavier_uniform_(self.weights[q])
        
    def reset_parameters_like_torch(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weights,gain=gain)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self,x):
        # Input to layer
        x = torch.cat([(x**i) for i in range(1,self.q+1)],dim=1)
        x = torch.nn.functional.conv2d(x,self.weights,bias=self.bias,padding=self.padding,dilation=self.dilation)        
        return x
