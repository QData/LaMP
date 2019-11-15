''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F
from lamp.Models import LAMP
from lamp.Beam import Beam
from pdb import set_trace as stop
import lamp.Constants as Constants
import numpy 

def get_attn_padding_mask(seq_q, seq_k, unsqueeze=True):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)   # bx1xsk
    if unsqueeze:
        pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask

def translate(model,opt,src_batch,adj):
    ''' Translation work in one batch '''
    tt = torch.cuda if opt.cuda else torch

    # Batch size is in different location depending on data.
    src_seq, src_pos = src_batch
    batch_size = src_seq.size(0)
    beam_size = opt.beam_size

    #- Enocde
    enc_output, *_ = model.encoder(src_seq, adj, src_pos)

    #--- Repeat data for beam
    src_seq = src_seq.data.repeat(1, beam_size).view(
            src_seq.size(0) * beam_size, src_seq.size(1))

    enc_output = enc_output.data.repeat(1, beam_size, 1).view(
            enc_output.size(0) * beam_size, enc_output.size(1), enc_output.size(2))

    #--- Prepare beams
    beams = [Beam(beam_size, opt.cuda) for _ in range(batch_size)]
    beam_inst_idx_map = {
        beam_idx: inst_idx for inst_idx, beam_idx in enumerate(range(batch_size))}
    n_remaining_sents = batch_size

    if opt.decoder == 'rnn_m':
        decoder_hidden = enc_output.mean(1)

    #- Decode
    for i in range(opt.max_token_seq_len_d):

        len_dec_seq = i + 1

        # -- Preparing decoded data seq -- #
        # size: batch x beam x seq
        
        dec_partial_seq = torch.stack([b.get_current_state() for b in beams if not b.done])

        # size: (batch * beam) x seq
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        dec_partial_seq = dec_partial_seq

        # -- Preparing decoded pos seq -- #
        # size: 1 x seq
        # dec_partial_pos = torch.arange(1, len_dec_seq + 1).unsqueeze(0)
        # size: (batch * beam) x seq
        # dec_partial_pos = dec_partial_pos.repeat(n_remaining_sents * beam_size, 1)
        # dec_partial_pos = dec_partial_pos.type(torch.LongTensor)

        if opt.cuda:
            dec_partial_seq = dec_partial_seq.cuda()
            # dec_partial_pos = dec_partial_pos.cuda()

        # -- Decoding -- #
        # print(dec_partial_seq) 
        if opt.decoder == 'rnn_m': 
            dec_enc_attn_pad_mask = get_attn_padding_mask(dec_partial_seq, src_seq,unsqueeze=False)
            dec_output, decoder_hidden, _ = model.decoder.forward_step(dec_partial_seq[:,-1].unsqueeze(1),decoder_hidden.squeeze(),enc_output,dec_enc_attn_pad_mask=dec_enc_attn_pad_mask)
            dec_output = dec_output[-1,:, :]
        else:
            # dec_output, *_ = model.decoder(dec_partial_seq, dec_partial_pos, src_seq, enc_output)
            dec_output, *_ = model.decoder(dec_partial_seq, src_seq, enc_output)
            dec_output = dec_output[:, -1, :] # (batch * beam) * d_model
            dec_output = model.tgt_word_proj(dec_output)

            # dec_output += model.U(enc_output.mean(1))#.unsqueeze(1)
            
       
        # Mask previously predicted labels
        for J in range(dec_output.size(0)):
            dec_output.data[J].index_fill_(0,dec_partial_seq.data[J], -float('inf'))

        out = F.log_softmax(dec_output,dim=1)
        

        # batch x beam x n_words
        word_lk = out.view(n_remaining_sents, beam_size, -1).contiguous()

        active_beam_idx_list = []
        for beam_idx in range(batch_size):
            if beams[beam_idx].done:
                continue

            inst_idx = beam_inst_idx_map[beam_idx]
            if not beams[beam_idx].advance(word_lk.data[inst_idx]):
                active_beam_idx_list += [beam_idx]

        if not active_beam_idx_list:
            # all instances have finished their path to <EOS>
            break


        # in this section, the sentences that are still active are
        # compacted so that the decoder is not run on completed sentences
        active_inst_idxs = tt.LongTensor([beam_inst_idx_map[k] for k in active_beam_idx_list])

        # update the idx mapping
        beam_inst_idx_map = {
            beam_idx: inst_idx for inst_idx, beam_idx in enumerate(active_beam_idx_list)}

        def update_active_seq(seq_var, active_inst_idxs):
            ''' Remove the src sequence of finished instances in one batch. '''

            inst_idx_dim_size, *rest_dim_sizes = seq_var.size()
            inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
            new_size = (inst_idx_dim_size, *rest_dim_sizes)

            # select the active instances in batch
            original_seq_data = seq_var.data.view(n_remaining_sents, -1)
            active_seq_data = original_seq_data.index_select(0, active_inst_idxs)
            active_seq_data = active_seq_data.view(*new_size)

            return active_seq_data

        def update_active_enc_info(enc_info_var, active_inst_idxs):
            ''' Remove the encoder outputs of finished instances in one batch. '''

            inst_idx_dim_size, *rest_dim_sizes = enc_info_var.size()
            inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
            new_size = (inst_idx_dim_size, *rest_dim_sizes)

            # select the active instances in batch
            original_enc_info_data = enc_info_var.data.view(n_remaining_sents, -1, opt.d_model)
            active_enc_info_data = original_enc_info_data.index_select(0, active_inst_idxs)
            active_enc_info_data = active_enc_info_data.view(*new_size)

            return active_enc_info_data


        src_seq = update_active_seq(src_seq, active_inst_idxs)
        enc_output = update_active_enc_info(enc_output, active_inst_idxs)

        if opt.decoder == 'rnn_m':
            decoder_hidden = update_active_enc_info(decoder_hidden.transpose(0,1), active_inst_idxs)
            decoder_hidden = decoder_hidden.transpose(0,1)

        #- update the remaining size
        n_remaining_sents = len(active_inst_idxs)

    #- Return useful information
    all_hyp, all_hyp_scores, all_scores = [], [], []
    n_best = opt.n_best
    for beam_idx in range(batch_size):
        scores, tail_idxs = beams[beam_idx].sort_scores()
        all_scores += [scores[:n_best]]
        
        hyps = [beams[beam_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
        all_hyp += [hyps]
        all_hyp_scores += [[torch.exp(i)[0] for i in beams[beam_idx].all_scores]]

    return all_hyp, all_hyp_scores#,all_scores