import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import models.Constants as Constants
# from models.Modules import BottleLinear as Linear
from models.Modules import XavierLinear as Linear
from models.Layers import EncoderLayer,DecoderLayer,AutoregressiveDecoderLayer
from models.Attention import Attention1
from models.Modules import ScaledDotProductAttention
from pdb import set_trace as stop
from models import utils



class MLPDiscriminator(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False, dropout=0.1,no_enc_pos_embedding=False,n_tgt_vocab=0):

        super(MLPDiscriminator, self).__init__()

        self.layer1 = nn.Linear(d_model+n_tgt_vocab,d_model+n_tgt_vocab)
        self.layer2 = nn.Linear(d_model+n_tgt_vocab,d_model+n_tgt_vocab)
        self.layer3 = nn.Linear(d_model+n_tgt_vocab,1)


    def forward(self,x_in,y_vec):
        x_vec = x_in.sum(1)
        out1 = F.relu(self.layer1(torch.cat((x_vec,y_vec),1)))
        out2 = F.relu(self.layer2(out1))
        out3 = self.layer3(out2)

        return out3

class SADiscriminator(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False, dropout=0.1,no_enc_pos_embedding=False,n_tgt_vocab=0):

        super(SADiscriminator, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.onehot = onehot

        if onehot:
            self.src_word_emb = nn.Embedding(n_src_vocab, n_src_vocab, padding_idx=Constants.PAD)
            self.src_word_emb.weight.data.fill_(0)
            self.src_word_emb.weight.data[1:,1:] = torch.eye(self.src_word_emb.weight.data[1:].size(0))
            self.conv = nn.Conv1d(9, 512, 16, stride=1, padding=8, dilation=1, groups=1, bias=True)
        else:
            self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        if no_enc_pos_embedding is False:
            self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
            self.position_enc.weight.data = utils.position_encoding_init(n_position, d_word_vec)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.b = nn.Linear(d_model,n_tgt_vocab)
        self.C1 = nn.Linear(n_tgt_vocab,n_tgt_vocab)
        self.c2 = nn.Linear(n_tgt_vocab,1)

    def forward(self, src_seq, adj, src_pos, y_vec,return_attns=False):
        enc_input = self.src_word_emb(src_seq)
        
        if self.onehot:
            enc_input = self.conv(enc_input.transpose(1,2)).transpose(1,2)[:,0:-1,:]

        if hasattr(self, 'position_enc'):
            enc_input += self.position_enc(src_pos)

        enc_outputs = []
        
        enc_output = enc_input
        enc_slf_attn_mask = utils.get_attn_padding_mask(src_seq, src_seq)

        if adj:
            enc_slf_attn_mask = enc_slf_attn_mask.type(torch.float32)
            for idx in range(enc_slf_attn_mask.size(0)):
                enc_slf_attn_mask[idx][0:adj[idx].size(0),0:adj[idx].size(0)] = swap_0_1(adj[idx],1,0)
            enc_slf_attn_mask = enc_slf_attn_mask.type(torch.uint8)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)

        # stop()
        # total_non_masked_words = src_seq.size(1) - enc_slf_attn_mask[:,0,:].sum(1)
        # x_vec = enc_output.sum(1)#/total_non_masked_words.unsqueeze(1)
        x_vec = enc_output.mean(1)

        local_score = (y_vec*self.b(x_vec)).sum(1)
        global_score = self.c2(F.relu(self.C1(y_vec))).view(-1)

        score = local_score + global_score

        return score

        
class SADiscriminator2(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False, dropout=0.1,no_enc_pos_embedding=False,n_tgt_vocab=0,label_adj_matrix=None):

        super(SADiscriminator2, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.onehot = onehot

        if onehot:
            self.src_word_emb = nn.Embedding(n_src_vocab, n_src_vocab, padding_idx=Constants.PAD)
            self.src_word_emb.weight.data.fill_(0)
            self.src_word_emb.weight.data[1:,1:] = torch.eye(self.src_word_emb.weight.data[1:].size(0))
            self.conv = nn.Conv1d(9, 512, 16, stride=1, padding=8, dilation=1, groups=1, bias=True)
        else:
            self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        if no_enc_pos_embedding is False:
            self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
            self.position_enc.weight.data = utils.position_encoding_init(n_position, d_word_vec)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])


        if label_adj_matrix is not None:
            for i in range(label_adj_matrix.size(0)):
                if label_adj_matrix[i].sum().item() < 1:
                    label_adj_matrix[i,i] = 1 #This prevents Nan output in attention (otherwise 0 attn weights occurs)
            
            self.label_adj_matrix = swap_0_1(label_adj_matrix,1,0).unsqueeze(0)
            
        else:
            self.label_adj_matrix = None


        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec)
        self.dropout = nn.Dropout(dropout)

        self.lab_layer_stack = nn.ModuleList()
        for _ in range(n_layers):
            self.lab_layer_stack.append(DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout,no_dec_self_att=False))
        self.constant_input = Variable(torch.from_numpy(np.arange(n_tgt_vocab)).view(-1,1))#.cuda()

        self.R1 = nn.Linear(d_model,d_model)
        self.R2 = nn.Linear(d_model,1)

        # self.R1 = nn.Linear(d_model, n_tgt_vocab, bias=True)


    def forward(self, src_seq, adj, src_pos, y_vec,return_attns=False):
        enc_input = self.src_word_emb(src_seq)
        batch_size = src_seq.size(0)
        
        if self.onehot:
            enc_input = self.conv(enc_input.transpose(1,2)).transpose(1,2)[:,0:-1,:]

        if hasattr(self, 'position_enc'):
            enc_input += self.position_enc(src_pos)

        enc_outputs = []
        
        if return_attns:
            enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = utils.get_attn_padding_mask(src_seq, src_seq)

        if adj:
            enc_slf_attn_mask = enc_slf_attn_mask.type(torch.float32)
            for idx in range(enc_slf_attn_mask.size(0)):
                enc_slf_attn_mask[idx][0:adj[idx].size(0),0:adj[idx].size(0)] = swap_0_1(adj[idx],1,0)

            enc_slf_attn_mask = enc_slf_attn_mask.type(torch.uint8)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)

        # x_vec = enc_output.sum(1)
        # stop()


        dec_input = self.tgt_word_emb(self.constant_input.repeat(1,batch_size).transpose(0,1).cuda())
        dec_input = dec_input*y_vec.unsqueeze(2)

        dec_enc_attn_pad_mask = utils.get_attn_padding_mask(self.constant_input.repeat(1,batch_size).transpose(0,1).cuda(), src_seq)


        if self.label_adj_matrix is not None:
            dec_slf_attn_mask = self.label_adj_matrix.repeat(batch_size,1,1).cuda().type(torch.uint8)
        else:
            # dec_slf_attn_mask = torch.eye(dec_input.size(1)).repeat(batch_size,1,1).cuda().byte()
            dec_slf_attn_mask = None


        dec_output = dec_input
        for dec_layer in self.lab_layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output,slf_attn_mask=dec_slf_attn_mask,dec_enc_attn_mask=dec_enc_attn_pad_mask)

        # stop()
        score = self.R2(F.relu(self.R1(dec_output))).sum(1)
        
        # seq_logit = self.R1(dec_output)
        # score = (seq_logit*Variable(torch.eye(seq_logit.size(1)),requires_grad=False).cuda().unsqueeze(0).repeat(batch_size,1,1)).sum(1)
        

        return score


class MatchingDiscriminator(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False, dropout=0.1,no_enc_pos_embedding=False,n_tgt_vocab=0,label_adj_matrix=None):

        super(MatchingDiscriminator, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.onehot = onehot

        if onehot:
            self.src_word_emb = nn.Embedding(n_src_vocab, n_src_vocab, padding_idx=Constants.PAD)
            self.src_word_emb.weight.data.fill_(0)
            self.src_word_emb.weight.data[1:,1:] = torch.eye(self.src_word_emb.weight.data[1:].size(0))
            self.conv = nn.Conv1d(9, 512, 16, stride=1, padding=8, dilation=1, groups=1, bias=True)
        else:
            self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        if no_enc_pos_embedding is False:
            self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
            self.position_enc.weight.data = utils.position_encoding_init(n_position, d_word_vec)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])


        if label_adj_matrix is not None:
            for i in range(label_adj_matrix.size(0)):
                if label_adj_matrix[i].sum().item() < 1:
                    label_adj_matrix[i,i] = 1 #This prevents Nan output in attention (otherwise 0 attn weights occurs)
            
            self.label_adj_matrix = swap_0_1(label_adj_matrix,1,0).unsqueeze(0)
            
        else:
            self.label_adj_matrix = None


        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec)
        self.dropout = nn.Dropout(dropout)

        self.lab_layer_stack = nn.ModuleList()
        for _ in range(n_layers):
            self.lab_layer_stack.append(DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout,no_dec_self_att=False))
        self.constant_input = Variable(torch.from_numpy(np.arange(n_tgt_vocab)).view(-1,1))#.cuda()

        self.R1 = nn.Linear(d_model,d_model)
        self.R2 = nn.Linear(d_model,1)

        # self.R1 = nn.Linear(d_model, n_tgt_vocab, bias=True)


    def forward(self, src_seq, adj, src_pos, y_vec,return_attns=False):
        enc_input = self.src_word_emb(src_seq)
        batch_size = src_seq.size(0)
        
        if self.onehot:
            enc_input = self.conv(enc_input.transpose(1,2)).transpose(1,2)[:,0:-1,:]

        if hasattr(self, 'position_enc'):
            enc_input += self.position_enc(src_pos)

        enc_outputs = []
        
        if return_attns:
            enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = utils.get_attn_padding_mask(src_seq, src_seq)

        if adj:
            enc_slf_attn_mask = enc_slf_attn_mask.type(torch.float32)
            for idx in range(enc_slf_attn_mask.size(0)):
                enc_slf_attn_mask[idx][0:adj[idx].size(0),0:adj[idx].size(0)] = swap_0_1(adj[idx],1,0)

            enc_slf_attn_mask = enc_slf_attn_mask.type(torch.uint8)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)

        # x_vec = enc_output.sum(1)
        # stop()


        dec_input = self.tgt_word_emb(self.constant_input.repeat(1,batch_size).transpose(0,1).cuda())
        dec_input = dec_input*y_vec.unsqueeze(2)

        dec_enc_attn_pad_mask = utils.get_attn_padding_mask(self.constant_input.repeat(1,batch_size).transpose(0,1).cuda(), src_seq)


        if self.label_adj_matrix is not None:
            dec_slf_attn_mask = self.label_adj_matrix.repeat(batch_size,1,1).cuda().type(torch.uint8)
        else:
            # dec_slf_attn_mask = torch.eye(dec_input.size(1)).repeat(batch_size,1,1).cuda().byte()
            dec_slf_attn_mask = None


        dec_output = dec_input
        for dec_layer in self.lab_layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output,slf_attn_mask=dec_slf_attn_mask,dec_enc_attn_mask=dec_enc_attn_pad_mask)


        score = self.R2(F.relu(self.R1(dec_output)))
        
        # seq_logit = self.R1(dec_output)
        # score = (seq_logit*Variable(torch.eye(seq_logit.size(1)),requires_grad=False).cuda().unsqueeze(0).repeat(batch_size,1,1)).sum(1)
        

        return score



class Discriminator(nn.Module):
    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_max_seq_e,n_max_seq_d, n_layers_enc=6,n_layers_dec=6,
            n_head=8,d_word_vec=512, d_model=512, d_inner_hid=1024, d_k=64, d_v=64,
            dropout=0.1, dec_dropout=0.1, proj_share_weight=True, embs_share_weight=True, 
            encoder='selfatt',decoder='sa_m',enc_transform='max',onehot=False,no_enc_pos_embedding=False,dec_reverse=False,no_dec_self_att=False,loss='ce',label_adj_matrix=None,label_mask=None):
        super(Discriminator, self).__init__()

        self.discriminator = SADiscriminator2(n_src_vocab, n_max_seq_e, n_layers=n_layers_enc, n_head=n_head,
                d_word_vec=d_word_vec, d_model=d_model,d_k=d_k, d_v=d_v,
                d_inner_hid=d_inner_hid, onehot=onehot, dropout=dropout,no_enc_pos_embedding=no_enc_pos_embedding,n_tgt_vocab=n_tgt_vocab)

    def forward(self, src,adj, tgt):
        return self.discriminator(src[0], adj, src[1],tgt)