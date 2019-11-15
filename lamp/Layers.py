''' Define the Layers '''
import torch.nn as nn
from lamp.SubLayers import MultiHeadAttention, PositionwiseFeedForward
from pdb import set_trace as stop
import math
import torch
from lamp import utils

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_input)
        
        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head,n_head2, d_k, d_v,dropout=0.1,dropout2=False,
        no_dec_self_att=False,ffn=True,attn_type='softmax'):
        super(DecoderLayer, self).__init__()
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn1 = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

        if not no_dec_self_att:
            self.slf_attn = MultiHeadAttention(n_head2, d_model, d_k, d_v, dropout=dropout,dropout2=dropout2)
        
        self.pos_ffn2 = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
        
    def forward(self, dec_input, enc_output,slf_attn_mask=None,dec_enc_attn_mask=None):
        dec_output, dec_enc_attn = self.enc_attn(dec_input, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn1(dec_output)

        if hasattr(self, 'slf_attn'):
            dec_output_int = dec_output
            dec_output, dec_slf_attn = self.slf_attn(dec_output, dec_output, dec_output, attn_mask=slf_attn_mask,dec_self=True)
        else:
            dec_slf_attn = None
            dec_output_int = None

        dec_output = self.pos_ffn2(dec_output)

        
        return dec_output, dec_output_int, dec_slf_attn, dec_enc_attn

























###########################################################################################




class AutoregressiveDecoderLayer(nn.Module):    
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1,no_dec_self_att=False,ffn=True):
        super(AutoregressiveDecoderLayer, self).__init__()
        self.slf_attn = False
        
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn1 = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn2 = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
        

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):

        dec_output, dec_enc_attn = self.enc_attn(dec_input, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn1(dec_output)

        dec_output, dec_slf_attn = self.slf_attn(dec_output, dec_output, dec_output, attn_mask=slf_attn_mask)
        dec_output = self.pos_ffn2(dec_output)

        return dec_output, dec_output_int, dec_slf_attn, dec_enc_attn


class GraphConvolution(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1,no_dec_self_att=False,ffn=True,bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = d_model
        self.out_features = d_model
        self.weight1 = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.weight2 = nn.Parameter(torch.FloatTensor(d_model, d_model))
        if bias:
            self.bias1 = nn.Parameter(torch.FloatTensor(d_model))
            self.bias2 = nn.Parameter(torch.FloatTensor(d_model))
        else:
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None,label_vec=None):
        stop()

        enc_dec_input = torch.cat((enc_output,dec_input),1)
        support = torch.bmm(enc_dec_input, self.weight.repeat(enc_dec_input.size(0),1,1))

        enc_dec_mask = torch.cat((dec_enc_attn_mask,torch.zeros(dec_input.size(1),dec_input.size(1)).cuda()))
        output = torch.bmm(slf_attn_mask.repeat(support.size(0),1,1), support)
        
        if self.bias1 is not None:
            output = output + self.bias1


        if slf_attn_mask is not None:
            slf_attn_mask = torch.zeros(dec_input.size(1),dec_input.size(1)).cuda()
        slf_attn_mask = utils.swap_0_1(slf_attn_mask,1,0)

        support = torch.bmm(dec_input, self.weight.repeat(dec_input.size(0),1,1))        
        output = torch.bmm(slf_attn_mask.repeat(support.size(0),1,1), support)
  
        if self.bias2 is not None:
            return output + self.bias2,None,None
        else:
            return output,None,None