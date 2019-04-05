import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.autograd import Variable
from pdb import set_trace as stop


class XavierLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=True):
        super(XavierLinear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal(self.linear.weight)
    def forward(self, x):
        return self.linear(x)

class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''
    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


# class Linear(nn.Module):
#     def __init__(self, d_in, d_out, bias=True):
#         super(Linear, self).__init__()
#         self.linear = nn.Linear(d_in, d_out, bias=bias)
#         init.xavier_normal(self.linear.weight)
#     def forward(self, x):
#         return self.linear(x)


# class BottleLinear(Bottle, Linear):
#     pass

class BottleSoftmax(Bottle, nn.Softmax):
    pass


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1, attn_type='softmax'):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        if attn_type == 'softmax':
            self.attn_type = nn.Softmax(dim=2)
            # self.softmax = BottleSoftmax()
        else:
            self.attn_type = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None,stop_sig=False):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -np.inf)

        if stop_sig:
            print('**')
            stop()


        attn = self.attn_type(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


