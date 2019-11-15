import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lamp.Constants as Constants
from lamp.Layers import EncoderLayer,DecoderLayer
from lamp.SubLayers import ScaledDotProductAttention
from lamp.SubLayers import PositionwiseFeedForward
from lamp.SubLayers import XavierLinear
from lamp.Encoders import MLPEncoder,GraphEncoder,RNNEncoder
from lamp.Decoders import MLPDecoder,RNNDecoder,GraphDecoder
from pdb import set_trace as stop 
from lamp import utils
import copy

 

class LAMP(nn.Module):
    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_max_seq_e,n_max_seq_d, n_layers_enc=6,n_layers_dec=6,
            n_head=8,n_head2=8,d_word_vec=512, d_model=512, d_inner_hid=1024, d_k=64, d_v=64,
            dropout=0.1, dec_dropout=0.1,dec_dropout2=0.1, proj_share_weight=True, embs_share_weight=True, 
            encoder='selfatt',decoder='sa_m',enc_transform='',onehot=False,no_enc_pos_embedding=False,
            no_dec_self_att=False,loss='ce',label_adj_matrix=None,label_mask=None,matching_mlp=False,
            graph_conv=False,attn_type='softmax',int_preds=False):

        super(LAMP, self).__init__()
        self.decoder_type = decoder
        self.onehot = onehot
        self.loss = loss
        
        self.enc_vec = False
        if encoder == 'mlp' or enc_transform != '':
            self.enc_vec = True
        
        ############# Encoder ###########
        if encoder == 'mlp':
            self.encoder = MLPEncoder(
                n_src_vocab, n_max_seq_e, n_layers=n_layers_enc, n_head=n_head,
                d_word_vec=d_word_vec, d_model=d_model,d_k=d_k, d_v=d_v,
                d_inner_hid=d_inner_hid,onehot=onehot, dropout=dropout)
        elif encoder == 'graph':
            self.encoder = GraphEncoder(
                n_src_vocab, n_max_seq_e, n_layers=n_layers_enc, n_head=n_head,
                d_word_vec=d_word_vec, d_model=d_model,d_k=d_k, d_v=d_v,
                d_inner_hid=d_inner_hid, onehot=onehot, dropout=dropout,
                no_enc_pos_embedding=no_enc_pos_embedding,enc_transform=enc_transform)
        elif encoder == 'rnn':
            self.encoder = RNNEncoder(
                n_src_vocab, n_max_seq_e, n_layers=n_layers_enc, n_head=n_head,
                d_word_vec=d_word_vec, d_model=d_model,d_k=d_k, d_v=d_v,
                d_inner_hid=d_inner_hid,onehot=onehot, dropout=dropout)
        else:
            raise NotImplementedError
        
        ############# Decoder ###########
        if decoder == 'rnn_m':
            self.decoder = RNNDecoder(
                n_tgt_vocab, n_max_seq_d, n_layers=n_layers_dec, n_head=n_head,
                d_word_vec=d_word_vec, d_model=d_model,d_k=d_k, d_v=d_v,
                d_inner_hid=d_inner_hid, dropout=dec_dropout)
        elif decoder == 'graph':
            self.decoder = GraphDecoder(
                n_tgt_vocab, n_max_seq_d, n_layers=n_layers_dec, n_head=n_head,
                n_head2=n_head2,d_word_vec=d_word_vec, d_model=d_model,d_k=d_k, d_v=d_v,
                d_inner_hid=d_inner_hid, dropout=dec_dropout,dropout2=dec_dropout2,
                no_dec_self_att=no_dec_self_att,label_adj_matrix=label_adj_matrix,
                label_mask=label_mask,enc_vec=self.enc_vec,graph_conv=graph_conv,
                attn_type=attn_type)
        elif decoder == 'mlp':
            self.decoder = MLPDecoder(
                n_tgt_vocab, n_max_seq_e, n_max_seq_d, n_layers=n_layers_dec, n_head=n_head,
                d_word_vec=d_word_vec, d_model=d_model,d_k=d_k, d_v=d_v,
                d_inner_hid=d_inner_hid, dropout=dec_dropout, enc_transform=enc_transform)
        else:
            raise NotImplementedError


        
        bias = False
        if self.decoder_type in ['mlp','graph','star'] and not proj_share_weight:
            bias = True


        assert d_model == d_word_vec

        if self.decoder_type != 'mlp':
            if proj_share_weight:
                self.tgt_word_proj = XavierLinear(d_model, n_tgt_vocab, bias=bias)
                self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight
            else:
                self.tgt_word_proj = XavierLinear(d_model, 1, bias=bias)
            if int_preds:
                self.tgt_word_proj_copy = XavierLinear(d_model, n_tgt_vocab, bias=bias)


    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        freezed_param_ids = set()
        if hasattr(self.encoder, 'position_enc'):
            enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
            freezed_param_ids = freezed_param_ids | enc_freezed_param_ids
        if self.onehot:
            enc_onehot_param_ids = set(map(id, self.encoder.src_word_emb.parameters()))
            freezed_param_ids = freezed_param_ids | enc_onehot_param_ids
    
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)


    def forward(self, src, adj, tgt_seq, binary_tgt,return_attns=False,int_preds=False):
        batch_size = src[0].size(0)
        src_seq, src_pos = src
        if self.decoder_type in ['sa_m','rnn_m']: 
            tgt_seq = tgt_seq[:, :-1]

        enc_output, *enc_self_attns = self.encoder(src_seq, adj, src_pos,return_attns=return_attns)
        dec_output, *dec_output2 = self.decoder(tgt_seq,src_seq,enc_output,return_attns=return_attns,int_preds=int_preds)

        if self.decoder_type == 'rnn_m':
            seq_logit = dec_output
        elif self.decoder_type == 'mlp':
            seq_logit = dec_output
        else:
            seq_logit = self.tgt_word_proj(dec_output)
            if self.decoder_type == 'graph':
                seq_logit = torch.diagonal(seq_logit,0,1,2)
        if int_preds:
            intermediate_preds = []
            tgt_word_proj_copy = self.tgt_word_proj.linear.weight.data.detach().repeat(batch_size,1,1)
            for int_idx,int_out in enumerate(dec_output2[0][:-1]):
                int_out = torch.bmm(int_out,tgt_word_proj_copy.transpose(1,2))
                intermediate_preds += [torch.diagonal(int_out,0,1,2)]
            return seq_logit.view(-1, seq_logit.size(-1)),enc_output, intermediate_preds
        elif return_attns:
            return seq_logit.view(-1,seq_logit.size(-1)),enc_output,enc_self_attns,dec_output2
        else:
            return seq_logit.view(-1,seq_logit.size(-1)),enc_output,None
