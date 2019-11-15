import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lamp.Constants as Constants
from lamp.Layers import EncoderLayer,DecoderLayer
from lamp.SubLayers import ScaledDotProductAttention
from lamp.SubLayers import PositionwiseFeedForward
from lamp.SubLayers import XavierLinear
from pdb import set_trace as stop 
from lamp import utils
import copy


 
class MLPEncoder(nn.Module):
    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False, dropout=0.1):
        super(MLPEncoder, self).__init__()
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.linear1 = nn.Linear(n_src_vocab,d_model)

    def forward(self, src_seq, adj, src_pos, return_attns=False):
        enc_output = self.linear1(src_seq)
        return enc_output.view(src_seq.size(0),1,-1),None



class GraphEncoder(nn.Module):
    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False,enc_transform='',
            dropout=0.1,no_enc_pos_embedding=False):

        super(GraphEncoder, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.onehot = onehot
        self.enc_transform = enc_transform
        self.dropout = nn.Dropout(dropout) 

        if onehot:
            self.src_word_emb = nn.Embedding(n_src_vocab, n_src_vocab, padding_idx=Constants.PAD)
            self.src_word_emb.weight.data.fill_(0)
            self.src_word_emb.weight.data[1:,1:] = torch.eye(self.src_word_emb.weight.data[1:].size(0))
            self.conv1 = nn.Conv1d(9, d_model, 16, stride=1, padding=8, dilation=1, groups=1, bias=True)
            self.conv2 = nn.Conv1d(d_model, d_model, 16, stride=1, padding=8, dilation=1, groups=1, bias=True)
        else:
            self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        if no_enc_pos_embedding is False:
            self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
            self.position_enc.weight.data = utils.position_encoding_init(n_position, d_word_vec)


        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, adj, src_pos, return_attns=False):
        batch_size = src_seq.size(0)
        enc_input = self.src_word_emb(src_seq)
        
        if self.onehot:
            enc_input = F.relu(self.dropout(self.conv1(enc_input.transpose(1,2))))[:,:,0:-1]
            enc_input = F.max_pool1d(enc_input,2,2)
            enc_input = F.relu(self.conv2(enc_input).transpose(1,2))[:,0:-1,:]
            enc_input += self.position_enc(src_pos[:,0:enc_input.size(1)])
            src_seq = src_seq[:,0:enc_input.size(1)]
        elif hasattr(self, 'position_enc'):
            enc_input += self.position_enc(src_pos)

        enc_outputs = []
        
        if return_attns: enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = utils.get_attn_padding_mask(src_seq, src_seq)


        if adj:
            enc_slf_attn_mask = enc_slf_attn_mask.type(torch.float32)
            for idx in range(len(adj)):
                enc_slf_attn_mask[idx][0:adj[idx].size(0),0:adj[idx].size(0)] = utils.swap_0_1(adj[idx],1,0)
            enc_slf_attn_mask = enc_slf_attn_mask.type(torch.uint8)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)

            if return_attns: enc_slf_attns += [enc_slf_attn]

        if self.enc_transform != '':
            if self.enc_transform == 'max':
                enc_output = F.max_pool1d(enc_output.transpose(1,2),x.size(1)).squeeze()
            elif self.enc_transform == 'sum':
                enc_output = enc_output.sum(1)
            elif self.enc_transform == 'mean':
                enc_output = enc_output.sum(1)/((src_seq > 0).sum(dim=1).float().view(-1,1))
            elif self.enc_transform == 'flatten':
                enc_output = enc_output.view(batch_size,-1).float()
            enc_output = enc_output.view(batch_size,1,-1)
        
        if return_attns:
            return enc_output,enc_slf_attns
        else:
            return enc_output,None

class RNNEncoder(nn.Module):
    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False, dropout=0.1):

        super(RNNEncoder, self).__init__()
        
        self.onehot = onehot

        if onehot:
            d_word_vec = 9
            self.src_word_emb = nn.Embedding(n_src_vocab, n_src_vocab, padding_idx=Constants.PAD)
            self.src_word_emb.weight.data.fill_(0)
            self.src_word_emb.weight.data[1:,1:] = torch.eye(self.src_word_emb.weight.data[1:].size(0))
            self.conv = nn.Conv1d(9, 512, 16, stride=1, padding=0, dilation=1, groups=1, bias=True)
        else:
            self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.brnn = nn.GRU(d_word_vec,d_model,n_layers,batch_first=True,bidirectional=True,dropout=dropout)
        self.U = nn.Linear(d_model*2,d_model)

    def forward(self, src_seq,adj, src_pos, return_attns=False):
        enc_input = self.src_word_emb(src_seq)
        enc_output,_ = self.brnn(enc_input)
        enc_output = self.U(enc_output)
        
        return enc_output,None