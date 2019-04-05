''' Handling the data io '''
import argparse
import torch
import transformer.Constants as Constants
from pdb import set_trace as stop


# python preprocess.py -train_src data/reuters/train_inputs.txt -train_tgt data/reuters/train_labels.txt -valid_src data/reuters/valid_inputs.txt -valid_tgt data/reuters/valid_labels.txt -test_src data/reuters/test_inputs.txt -test_tgt data/reuters/test_labels.txt -save_data data/reuters/train_valid_test.pt -max_seq_len 300

# python preprocess.py -train_src data/rcv1/train_inputs.txt -train_tgt data/rcv1/train_labels.txt -valid_src data/rcv1/valid_inputs.txt -valid_tgt data/rcv1/valid_labels.txt -test_src data/rcv1/test_inputs.txt -test_tgt data/rcv1/test_labels.txt -save_data data/rcv1/train_valid_test.pt -max_seq_len 300


# python preprocess.py -train_src data/gm12878/train_inputs.txt -train_tgt data/gm12878/train_labels.txt -valid_src data/gm12878/valid_inputs.txt -valid_tgt data/gm12878/valid_labels.txt -test_src data/gm12878/test_inputs.txt -test_tgt data/gm12878/test_labels.txt -save_data data/gm12878/train_valid_test.pt -max_seq_len 200

# python preprocess.py -train_src data/bookmarks/train_inputs.txt -train_tgt data/bookmarks/train_labels.txt -valid_src data/bookmarks/valid_inputs.txt -valid_tgt data/bookmarks/valid_labels.txt -test_src data/bookmarks/test_inputs.txt -test_tgt data/bookmarks/test_labels.txt -save_data data/bookmarks/train_valid_test.pt -max_seq_len 500

# python preprocess.py -train_src data/bibtext/train_inputs.txt -train_tgt data/bibtext/train_labels.txt -valid_src data/bibtext/valid_inputs.txt -valid_tgt data/bibtext/valid_labels.txt -test_src data/bibtext/test_inputs.txt -test_tgt data/bibtext/test_labels.txt -save_data data/bibtext/train_valid_test.pt -max_seq_len 500

# python preprocess.py -train_src data/delicious/train_inputs.txt -train_tgt data/delicious/train_labels.txt -valid_src data/delicious/valid_inputs.txt -valid_tgt data/delicious/valid_labels.txt -test_src data/delicious/test_inputs.txt -test_tgt data/delicious/test_labels.txt -save_data data/delicious/train_valid_test.pt -max_seq_len 500

# python preprocess.py -train_src data/delicious2/train_inputs.txt -train_tgt data/delicious2/train_labels.txt -valid_src data/delicious2/valid_inputs.txt -valid_tgt data/delicious2/valid_labels.txt -test_src data/delicious2/test_inputs.txt -test_tgt data/delicious2/test_labels.txt -save_data data/delicious2/train_valid_test.pt -max_seq_len 500

# python preprocess.py -train_src data/sider/train_inputs.txt -train_adj data/sider/train_matrices.txt -train_tgt data/sider/train_labels.txt -valid_src data/sider/valid_inputs.txt -valid_adj data/sider/valid_matrices.txt -valid_tgt data/sider/valid_labels.txt -test_src data/sider/test_inputs.txt -test_adj data/sider/test_matrices.txt -test_tgt data/sider/test_labels.txt -save_data data/sider/train_valid_test.pt -max_seq_len 915


def read_adj_matrix(inst_file):

    word_insts = []
    with open(inst_file) as f:
        for sent in f:
            words = sent.split()
            word_inst  = list(map(int, words))
            word_insts += [word_inst]

    return word_insts


def read_instances_from_file(inst_file, max_sent_len, keep_case, use_bos_eos=True):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            words = sent.split()
            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            word_inst = words[:max_sent_len]

            if word_inst:
                if use_bos_eos:
                    word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
                else:
                    word_insts += [word_inst]
            else:
                word_insts += [None]

    # print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts

def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    '''Word mapping to idx'''
    try:
        word_insts = ['</s>' if x is None else x for x in word_insts]
        return [[word2idx[w] if w in word2idx else Constants.UNK for w in s] for s in word_insts]
    except:

        print('error in preprocess.py')
        stop()

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_adj', required=False)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-valid_src', required=True)
    parser.add_argument('-valid_adj', required=False)
    parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-test_src', required=True)
    parser.add_argument('-test_adj', required=False)
    parser.add_argument('-test_tgt', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-max_seq_len', type=int, default=300)
    parser.add_argument('-max_tgt_len', type=int, default=100000)
    parser.add_argument('-min_word_count', type=int, default=0)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)


    opt = parser.parse_args()
    opt.max_seq_len = opt.max_seq_len + 2 # include the <s> and </s>


    # Training set
    train_tgt_word_insts = read_instances_from_file(opt.train_tgt, opt.max_tgt_len, opt.keep_case)
    if opt.train_adj:
        train_src_word_insts = read_instances_from_file(opt.train_src, opt.max_seq_len, opt.keep_case,use_bos_eos=False)
        train_adj_insts = read_adj_matrix(opt.train_adj)
    else:
        train_src_word_insts = read_instances_from_file(opt.train_src, opt.max_seq_len, opt.keep_case)

    

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # Validation set
    
    valid_tgt_word_insts = read_instances_from_file(opt.valid_tgt, opt.max_tgt_len, opt.keep_case)
    if opt.valid_adj:
        valid_src_word_insts = read_instances_from_file(opt.valid_src, opt.max_seq_len, opt.keep_case,use_bos_eos=False)
        valid_adj_insts = read_adj_matrix(opt.valid_adj)
    else:
        valid_src_word_insts = read_instances_from_file(opt.valid_src, opt.max_seq_len, opt.keep_case)

    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))

    # Build vocabulary
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)


    if opt.test_adj:
        test_adj_insts = read_adj_matrix(opt.test_adj)
        test_src_word_insts=read_instances_from_file(opt.test_src,opt.max_seq_len,opt.keep_case,use_bos_eos=False)
    else:
        test_src_word_insts=read_instances_from_file(opt.test_src,opt.max_seq_len,opt.keep_case)

    
    test_src_insts=convert_instance_to_idx_seq(test_src_word_insts, src_word2idx)
    test_tgt_word_insts=read_instances_from_file(opt.test_tgt,opt.max_tgt_len,opt.keep_case)
    test_tgt_insts=convert_instance_to_idx_seq(test_tgt_word_insts, tgt_word2idx)

    if opt.train_adj:
        data = {
            'settings': opt,
            'dict': {
                'src': src_word2idx,
                'tgt': tgt_word2idx},
            'train': {
                'src': train_src_insts,
                'adj': train_adj_insts,
                'tgt': train_tgt_insts},
            'valid': {
                'src': valid_src_insts,
                'adj': valid_adj_insts,
                'tgt': valid_tgt_insts},
            'test': {
                'src': test_src_insts,
                'adj': test_adj_insts,
                'tgt': test_tgt_insts}}
    else:
        data = {
            'settings': opt,
            'dict': {
                'src': src_word2idx,
                'tgt': tgt_word2idx},
            'train': {
                'src': train_src_insts,
                'tgt': train_tgt_insts},
            'valid': {
                'src': valid_src_insts,
                'tgt': valid_tgt_insts},
            'test': {
                'src': test_src_insts,
                'tgt': test_tgt_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)


if __name__ == '__main__':
    main()
