''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
import math
import transformer.Constants as Constants
from pdb import set_trace as stop

class DataLoader(object):
    ''' For data iteration '''

    def __init__(
            self, src_word2idx, tgt_word2idx,
            src_insts=None, adj_insts=None, tgt_insts=None,
            cuda=True, batch_size=64, shuffle=True, test=False,
            max_encoder_len=1000000,drop_last=False):

        assert src_insts
        assert len(src_insts) >= batch_size



        if tgt_insts:
            assert len(src_insts) == len(tgt_insts)
        if adj_insts:
            assert len(src_insts) == len(adj_insts)
            self._adj_insts = adj_insts
        else:
            self._adj_insts = None


        self.cuda = cuda
        self.test = test
        self._n_batch = int(np.ceil(len(src_insts) / batch_size))
        if drop_last:
            self._n_batch -= 1

        self._batch_size = batch_size

        self._src_insts = src_insts
        
        self._tgt_insts = tgt_insts


        if src_word2idx is not None:
            src_idx2word = {idx:word for word, idx in src_word2idx.items()}
            self._src_word2idx = src_word2idx
            self._src_idx2word = src_idx2word
            self.long_input = True
        else:
            self._src_word2idx = src_insts[0]
            self.long_input = False

        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}
        
        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word

        self._iter_count = 0

        self.max_encoder_len = max_encoder_len

        self._need_shuffle = shuffle

        if self._need_shuffle:
            self.shuffle()

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def shuffle(self):
        ''' Shuffle data for a brand new start '''
        if self._tgt_insts and self._adj_insts:
            paired_insts = list(zip(self._src_insts, self._adj_insts,self._tgt_insts))
            random.shuffle(paired_insts)
            self._src_insts, self._adj_insts, self._tgt_insts = zip(*paired_insts)
        elif self._tgt_insts:
            paired_insts = list(zip(self._src_insts,self._tgt_insts))
            random.shuffle(paired_insts)
            self._src_insts, self._tgt_insts = zip(*paired_insts)
        else:
            random.shuffle(self._src_insts)


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def convert_string_to_mat(adj_string):
            dim = int(math.sqrt(len(adj_string)))

            output = torch.Tensor(adj_string).view(dim,dim)#.type(torch.uint8)

            if self.cuda:
                output = output.cuda()

            return(output)

        def construct_adj_mat(insts,encoder=False):

            inst_data_tensor = [convert_string_to_mat(inst) for inst in insts]

            return inst_data_tensor

        def pad_to_longest(insts,encoder=False):
            ''' Pad the instance to the max seq length in batch '''
            # stop()
            max_len = max(len(inst) for inst in insts)

            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst))
                for inst in insts])

            inst_position = np.array([
                [pos_i+1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(inst)]
                for inst in inst_data])

        
            inst_data_tensor = torch.LongTensor(inst_data)
            inst_position_tensor = torch.LongTensor(inst_position)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
                inst_position_tensor = inst_position_tensor.cuda()
            return inst_data_tensor, inst_position_tensor

        if self._iter_count < self._n_batch:

            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            src_insts = self._src_insts[start_idx:end_idx]

            if self._adj_insts:
                adj_insts = construct_adj_mat(self._adj_insts[start_idx:end_idx])
            else:
                adj_insts = None

            src_data, src_pos = pad_to_longest(src_insts,encoder=True)

            if not self.long_input:
                src_data = src_data.float()


            if not self._tgt_insts:
                return src_data, src_pos
            else:
                tgt_insts = self._tgt_insts[start_idx:end_idx]
                tgt_data, tgt_pos = pad_to_longest(tgt_insts)
                return (src_data, src_pos), (adj_insts), (tgt_data, tgt_pos)

        else:

            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()
