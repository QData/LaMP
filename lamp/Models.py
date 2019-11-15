import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lamp.Constants as Constants
# from lamp.Modules import BottleLinear as Linear
from lamp.Modules import XavierLinear as Linear
from lamp.Layers import EncoderLayer,DecoderLayer,AutoregressiveDecoderLayer,GraphConvolution
from lamp.Modules import ScaledDotProductAttention
from lamp.SubLayers import PositionwiseFeedForward
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
        self.onehot = onehot
        self.dropout = nn.Dropout(dropout)

        # self.linear1 = nn.Linear(n_src_vocab,int(n_src_vocab/4))
        # self.linear2 = nn.Linear(int(n_src_vocab/4),int(n_src_vocab/8))
        # self.linear3 = nn.Linear(int(n_src_vocab/8),d_model)

        self.linear1 = nn.Linear(n_src_vocab,d_model)


    def forward(self, src_seq, adj, src_pos, return_attns=False):

        # out1 = self.dropout(F.relu(self.linear1(src_seq)))
        # out2 = self.dropout(F.relu(self.linear2(out1)))
        # enc_output = self.dropout(self.linear3(out2))

        enc_output = self.linear1(src_seq)
        
        return enc_output.view(src_seq.size(0),1,-1),None


class EmbEncoder(nn.Module):
    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False, enc_transform='',dropout=0.1):
        super(EmbEncoder, self).__init__()
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

        # self.w_1 = nn.Linear(d_model,d_model)
        # self.w_2 = nn.Linear(d_model,d_model)

        

    def forward(self, src_seq, adj, src_pos, return_attns=False):
        batch_size = src_seq.size(0)
        
        enc_output = self.src_word_emb(src_seq)
        # stop()

        if self.onehot:
            enc_output = F.relu(self.dropout(self.conv1(enc_output.transpose(1,2)).transpose(1,2)))
            enc_output = F.relu(self.conv1(enc_output.transpose(1,2)).transpose(1,2))
            enc_output = enc_output[:,0:src_seq.size(1),:]


        if self.enc_transform == 'max':
            enc_output = F.max_pool1d(enc_output.transpose(1,2),x.size(1)).squeeze()
            enc_output = enc_output.view(batch_size,1,-1)
        elif self.enc_transform == 'sum':

            enc_output = enc_output.sum(1)
            
            # enc_output = F.relu(self.w_1(enc_output))
            # enc_output = self.dropout(self.w_2(enc_output))

            enc_output = enc_output.view(batch_size,1,-1)


        elif self.enc_transform == 'mean':
            enc_output = enc_output.sum(1)/((src_seq > 0).sum(dim=1).float().view(-1,1))
            enc_output = enc_output.view(batch_size,1,-1)
        elif self.enc_transform == 'flatten':
            enc_output = enc_output.view(batch_size,-1).float()
            enc_output = enc_output.view(batch_size,1,-1)
        else:
            pass

        
        
        return enc_output,None


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
            # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
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
            # stop()
            enc_input = F.relu(self.dropout(self.conv1(enc_input.transpose(1,2))))[:,:,0:-1]

            enc_input = F.max_pool1d(enc_input,2,2)
            enc_input = F.relu(self.conv2(enc_input).transpose(1,2))[:,0:-1,:]

            # enc_input = self.conv(enc_input.transpose(1,2)).transpose(1,2)[:,0:-1,:]
            # stop()
            enc_input += self.position_enc(src_pos[:,0:enc_input.size(1)])
            src_seq = src_seq[:,0:enc_input.size(1)]
        elif hasattr(self, 'position_enc'):
            enc_input += self.position_enc(src_pos)

        enc_outputs = []
        
        if return_attns: enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = utils.get_attn_padding_mask(src_seq, src_seq)

        # stop()

        if adj:
            enc_slf_attn_mask = enc_slf_attn_mask.type(torch.float32)
            for idx in range(len(adj)):
                enc_slf_attn_mask[idx][0:adj[idx].size(0),0:adj[idx].size(0)] = utils.swap_0_1(adj[idx],1,0)
            enc_slf_attn_mask = enc_slf_attn_mask.type(torch.uint8)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)

            # enc_outputs += [enc_output]
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
        # if self.onehot:
        #     enc_output = self.conv(enc_output.transpose(1,2)).transpose(1,2)
        enc_output,_ = self.brnn(enc_input)

        enc_output = self.U(enc_output)
        
        return enc_output,None

#################################################################################################################
#################################################################################################################

class SADecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1, dec_type='sa_m'):
        # dropout = 0

        super(SADecoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.dec_type = dec_type

        # self.position_enc = nn.Embedding(
        #     n_position, d_word_vec, padding_idx=Constants.PAD)
        # self.position_enc.weight.data = utils.position_encoding_init(n_position, d_word_vec)

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            AutoregressiveDecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, src_seq, enc_output,return_attns=False,int_preds=False):
        # Word embedding look up
        dec_input = self.tgt_word_emb(tgt_seq)

        # Decode
        dec_slf_attn_pad_mask = utils.get_attn_padding_mask(tgt_seq, tgt_seq)

        dec_slf_attn_sub_mask = utils.get_attn_subsequent_mask(tgt_seq)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)

        dec_enc_attn_pad_mask = utils.get_attn_padding_mask(tgt_seq, src_seq)

        if return_attns:
            dec_slf_attns, dec_enc_attns = [], [] 

        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_output_int, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask)

            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]


        if return_attns:
            return dec_output, dec_slf_attns, dec_enc_attns
        else:
            return dec_output,


class RNNDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):
        # dropout = 0

        super(RNNDecoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.n_tgt_vocab = n_tgt_vocab


        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.dropout = nn.Dropout(dropout)

        self.attention_stack = nn.ModuleList([ScaledDotProductAttention(d_model,dropout=dropout)for _ in range(n_layers)])
        

        self.rnn_layer_stack = nn.ModuleList([
            nn.GRU(d_model+d_word_vec,d_model,batch_first=True,dropout=dropout)
            for _ in range(n_layers)])

        self.U = nn.Linear(self.d_model, self.n_tgt_vocab)
        self.V = nn.Linear(self.d_model, self.n_tgt_vocab)
        self.C = nn.Linear(self.d_model, self.n_tgt_vocab)

    def forward_step(self, input_var, decoder_hidden, encoder_outputs,dec_enc_attn_pad_mask=None):
        batch_size = input_var.size(0)
        
        embedded = self.tgt_word_emb(input_var)


        decoder_hidden = decoder_hidden.view(batch_size,1,-1)
        
        if encoder_outputs.size(1) == 1:
            dec_enc_attn_pad_mask=None

        for idx,dec_layer in enumerate(self.rnn_layer_stack):
            # print(decoder_hidden)''

            context,attn = self.attention_stack[idx](decoder_hidden.view(batch_size,1,-1),encoder_outputs,encoder_outputs,dec_enc_attn_pad_mask)
            rnn_input = torch.cat((embedded,context),2)
            embedded,decoder_hidden = dec_layer(rnn_input, decoder_hidden.view(1,batch_size,-1))

        output = self.U(decoder_hidden)
        output += self.V(embedded.view(batch_size,-1))
        output += self.C(context.view(batch_size,-1))

        return output, decoder_hidden, attn

    def forward(self, tgt_seq, src_seq, enc_output,return_attns=False,int_preds=False):
        # Word embedding look up
        # dec_input = self.tgt_word_emb(tgt_seq)[:,0,:]
      
        batch_size = enc_output.size(0)
        
        dec_enc_attn_pad_mask = utils.get_attn_padding_mask(tgt_seq, src_seq,unsqueeze=False)

        dec_output = torch.zeros(tgt_seq.size(0),tgt_seq.size(1),self.n_tgt_vocab).cuda()

        dec_input = tgt_seq[:,0].unsqueeze(1)

        decoder_hidden = enc_output.mean(1)


        for di in range(tgt_seq.size(1)):
            decoder_output,decoder_hidden,step_attn=self.forward_step(dec_input,decoder_hidden,enc_output,dec_enc_attn_pad_mask)

            dec_output[:,di,:] = decoder_output
            dec_input = F.log_softmax(decoder_output.view(batch_size,-1),dim=1).topk(1)[1].view(batch_size,-1)


        return dec_output,


class MLPDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, n_tgt_vocab, n_max_seq_e, n_max_seq_d, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1,enc_transform='mean'):
        # dropout = 0

        super(MLPDecoder, self).__init__()
        self.n_max_seq = n_max_seq_e
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.enc_transform = enc_transform

        if enc_transform in ['flatten']:
            raise NotImplementedError

        # self.linear1 = nn.Linear(d_model,d_model)
        # self.linear2 = nn.Linear(d_model,d_model)
        # self.linear3 = nn.Linear(d_model,d_model)
        # self.linear4 = nn.Linear(d_model,n_tgt_vocab)
        
        self.linear1 = nn.Linear(d_model,d_model)
        self.linear4 = nn.Linear(d_model,n_tgt_vocab)

        

    def forward(self, tgt_seq, src_seq, enc_output,return_attns=False,int_preds=False):
        batch_size = src_seq.size(0)
        x = enc_output.float()

        # out1 = self.dropout(F.relu(self.linear1(x)))
        # out2 = self.dropout(F.relu(self.linear2(out1)))
        # out3 = self.dropout(F.relu(self.linear3(out2)))
        # dec_dropout = self.linear4(out3)


        out1 = self.dropout(F.relu(self.linear1(x)))
        dec_dropout = self.linear4(out1)
        
        return dec_dropout.view(batch_size,1,-1),
    

class GraphDecoder(nn.Module):
    def __init__(
            self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8,n_head2=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1,dropout2=0.1,
            no_dec_self_att=False,label_adj_matrix=None,label_mask=None,
            enc_vec=True,graph_conv=False,attn_type='softmax'):
        super(GraphDecoder, self).__init__()
        self.enc_vec = enc_vec
        self.dropout = nn.Dropout(dropout)
        self.constant_input = torch.from_numpy(np.arange(n_tgt_vocab)).view(-1,1)

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec)
        
        if label_adj_matrix is not None:
            for i in range(label_adj_matrix.size(0)):
                if label_adj_matrix[i].sum().item() < 1:
                    label_adj_matrix[i,i] = 1 #This prevents Nan output in attention (otherwise 0 attn weights occurs)
            self.label_adj_matrix = utils.swap_0_1(label_adj_matrix,1,0).unsqueeze(0)
        else:
            if label_mask == 'eye':
                self.label_adj_matrix = torch.eye(n_tgt_vocab)
            elif label_mask == 'inveye':
                self.label_adj_matrix = 1-torch.eye(n_tgt_vocab)
            elif label_mask == 'none':
                self.label_adj_matrix = None
            else:
                NotImplementedError
        
        self.layer_stack = nn.ModuleList()
        for _ in range(n_layers):
            if graph_conv:
                 self.layer_stack.append(GraphConvolution(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout,no_dec_self_att=no_dec_self_att))
            else:
                self.layer_stack.append(DecoderLayer(d_model, d_inner_hid, n_head,n_head2, d_k, d_v, dropout=dropout,dropout2=dropout2,no_dec_self_att=no_dec_self_att,attn_type=attn_type))           


    def forward(self, tgt, src_seq, enc_output,return_attns=False, int_preds=False):
        batch_size = src_seq.size(0)
        if int_preds: int_outs = []
        if return_attns: dec_slf_attns, dec_enc_attns = [], []

        tgt_seq = self.constant_input.repeat(1,batch_size).transpose(0,1).cuda()

        dec_input = self.tgt_word_emb(tgt_seq)

        dec_enc_attn_pad_mask = None
        if not self.enc_vec:
            # stop()
            dec_enc_attn_pad_mask = utils.get_attn_padding_mask(tgt_seq, src_seq[:,0:enc_output.size(1)])

        if self.label_adj_matrix is not None:
            dec_slf_attn_mask = self.label_adj_matrix.repeat(batch_size,1,1).cuda().byte()
        else:
            dec_slf_attn_mask = None
            
        dec_output = dec_input
        for idx,dec_layer in enumerate(self.layer_stack):
            dec_output, dec_output_int, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output,slf_attn_mask=dec_slf_attn_mask,dec_enc_attn_mask=dec_enc_attn_pad_mask)

            if int_preds:
                if dec_output_int is not None:
                    int_outs += [dec_output_int]
                int_outs += [dec_output]
                

            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]


        if int_preds:
            return dec_output, int_outs
        elif return_attns:
            return dec_output, dec_slf_attns, dec_enc_attns
        else:
            return dec_output, None         



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
        elif encoder == 'emb':
            self.encoder = EmbEncoder(
                n_src_vocab, n_max_seq_e, n_layers=n_layers_enc, n_head=n_head,
                d_word_vec=d_word_vec, d_model=d_model,d_k=d_k, d_v=d_v,
                d_inner_hid=d_inner_hid,onehot=onehot, dropout=dropout,enc_transform=enc_transform)
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
        if decoder in ['sa_m','sa_b']:
            if decoder == 'sa_b':
                n_tgt_vocab = 2
            self.decoder = SADecoder(
                n_tgt_vocab, n_max_seq_d, n_layers=n_layers_dec, n_head=n_head,
                d_word_vec=d_word_vec, d_model=d_model,d_k=d_k, d_v=d_v,
                d_inner_hid=d_inner_hid, dropout=dec_dropout,dec_type=decoder)
        elif decoder == 'rnn_m':
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

        self.matching_mlp = False
        if matching_mlp:
            self.matching_mlp = True
            self.matching_network = MatchingGNN(n_src_vocab,n_tgt_vocab, n_max_seq_e, n_layers=n_layers_dec, n_head=n_head,
                n_head2=n_head2,d_word_vec=d_word_vec, d_model=d_model,d_k=d_k, d_v=d_v,
                d_inner_hid=d_inner_hid, dropout=dec_dropout,dropout2=dec_dropout2,
                no_dec_self_att=no_dec_self_att,label_adj_matrix=label_adj_matrix,onehot=onehot,
                label_mask=label_mask,enc_vec=self.enc_vec,graph_conv=graph_conv,no_enc_pos_embedding=no_enc_pos_embedding,
                attn_type=attn_type)
        
        bias = False
        if self.decoder_type in ['mlp','graph','star'] and not proj_share_weight:
            bias = True


        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module output shall be the same.'

        if self.decoder_type != 'mlp':
            if proj_share_weight:
                self.tgt_word_proj = Linear(d_model, n_tgt_vocab, bias=bias)
                self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight
            else:
                self.tgt_word_proj = Linear(d_model, 1, bias=bias)
            if int_preds:
                self.tgt_word_proj_copy = Linear(d_model, n_tgt_vocab, bias=bias)
            # self.tgt_layer_stack = nn.ModuleList()
            # for i in range(n_layers_dec-1):
            #   self.tgt_layer_stack.append(Linear(d_model, n_tgt_vocab, bias=bias))
            #   self.tgt_layer_stack.append(Linear(d_model, 1, bias=bias))


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

        if self.decoder_type in ['sa_m','rnn_m']: tgt_seq = tgt_seq[:, :-1]

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
        if self.matching_mlp and (binary_tgt is not None):
            # match_pred = self.matching_network(src_seq, adj, src_pos, F.sigmoid(seq_logit))
            match_pred = dec_output#.sum(2)
            match_true,int_true = self.matching_network(src_seq, adj, src_pos, binary_tgt)
            # match_true = self.matching_network(src_seq, adj, src_pos, binary_tgt,shared_embedding=self.decoder.tgt_word_emb)
            return seq_logit.view(-1, seq_logit.size(-1)),enc_output, [dec_output2[0],int_true]
        elif int_preds:
            intermediate_preds = []
            tgt_word_proj_copy = self.tgt_word_proj.linear.weight.data.detach().repeat(batch_size,1,1)
            for int_idx,int_out in enumerate(dec_output2[0][:-1]):
                # intermediate_preds += [torch.diagonal(self.tgt_layer_stack[int_idx](int_out),0,1,2)]
                int_out = torch.bmm(int_out,tgt_word_proj_copy.transpose(1,2))
                intermediate_preds += [torch.diagonal(int_out,0,1,2)]
            return seq_logit.view(-1, seq_logit.size(-1)),enc_output, intermediate_preds
        elif return_attns:
            return seq_logit.view(-1,seq_logit.size(-1)),enc_output,enc_self_attns,dec_output2
        else:
            return seq_logit.view(-1,seq_logit.size(-1)),enc_output,None
