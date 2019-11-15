import torch.nn as nn
import torch
import csv
from pdb import set_trace as stop
import numpy as np
import scipy
from torch import functional as F
import lamp.Constants as Constants


labels105u = ['go0000027','go0000067','go0000070','go0000071','go0000074','go0000082','go0000086','go0000087','go0000122','go0000154','go0000278','go0000280','go0000282','go0000283','go0000398','go0000902','go0000910','go0001403','go0006066','go0006259','go0006260','go0006261','go0006270','go0006281','go0006289','go0006298','go0006310','go0006319','go0006325','go0006338','go0006347','go0006348','go0006351','go0006355','go0006357','go0006360','go0006364','go0006365','go0006366','go0006367','go0006368','go0006396','go0006397','go0006402','go0006412','go0006413','go0006414','go0006457','go0006461','go0006464','go0006468','go0006487','go0006508','go0006511','go0006513','go0006605','go0006607','go0006608','go0006609','go0006610','go0006611','go0006623','go0006796','go0006886','go0006887','go0006888','go0006891','go0006893','go0006897','go0006906','go0006950','go0006970','go0006974','go0006979','go0006997','go0006999','go0007001','go0007010','go0007015','go0007020','go0007046','go0007049','go0007059','go0007067','go0007124','go0007126','go0007131','go0008283','go0015031','go0015986','go0016070','go0016071','go0016072','go0016192','go0016568','go0016573','go0019538','go0030036','go0030163','go0030490','go0042254','go0045045','go0045449','go0045944','go0048193']

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)

        if len(mask) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        self.true_dist = true_dist
        return self.criterion(x, true_dist)


def get_criterion(opt):
    if opt.binary_relevance:
        return nn.BCELoss(size_average=False)#weight=ranking_values)
    if opt.label_smoothing >0 :
        return LabelSmoothing(opt.tgt_vocab_size, Constants.PAD, opt.label_smoothing)
    else:
        weight = torch.ones(opt.tgt_vocab_size)
        weight[Constants.PAD] = 0
    return nn.CrossEntropyLoss(weight, size_average=False)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_pairwise_adj_rcv1(data_dict,interactions_file_name):

    interactions_file = csv.DictReader(open(interactions_file_name),delimiter='\t')

    interactions_dict = {}
    for line in interactions_file:
        node1 = line['#node1'].lower()
        node2 = line['node2'].lower()

        if not node2 in interactions_dict:
            interactions_dict[node2] = []
        interactions_dict[node2].append(node1)
        
    adjacency_matrix = torch.zeros((len(data_dict)-4,len(data_dict)-4))

    for tf_name1 in data_dict.keys():
        for tf_name2 in data_dict.keys():
            if tf_name1 != tf_name2:
                tf_root1 = tf_name1.split('_')[0]
                tf_root2 = tf_name2.split('_')[0]

                if tf_root1 in interactions_dict:
                    if tf_root2 in interactions_dict[tf_root1]:
                        adjacency_matrix[int(data_dict[tf_name1])-4,int(data_dict[tf_name2])-4] = 1
                        adjacency_matrix[int(data_dict[tf_name2])-4,int(data_dict[tf_name1])-4] = 1

                # if tf_root2 in interactions_dict:
                #     if tf_root1 in interactions_dict[tf_root2]:
                #         adjacency_matrix[int(data_dict[tf_name1])-4,int(data_dict[tf_name2])-4] = 1


    return adjacency_matrix

def get_pairwise_adj(data_dict,interactions_file_name):

    interactions_file = csv.DictReader(open(interactions_file_name),delimiter='\t')

    interactions_dict = {}
    for line in interactions_file:
        node1 = line['#node1'].lower()
        node2 = line['node2'].lower()

        if not node1 in interactions_dict:
            interactions_dict[node1] = []
        interactions_dict[node1].append(node2)

        if not node2 in interactions_dict:
            interactions_dict[node2] = []
        interactions_dict[node2].append(node1)
            
        
    adjacency_matrix = torch.zeros((len(data_dict)-4,len(data_dict)-4))

    for tf_name1 in data_dict.keys():
        for tf_name2 in data_dict.keys():
            if tf_name1 != tf_name2:
                tf_root1 = tf_name1.split('_')[0]
                tf_root2 = tf_name2.split('_')[0]

                if tf_root1 in interactions_dict:
                    if tf_root2 in interactions_dict[tf_root1]:
                        adjacency_matrix[int(data_dict[tf_name1])-4,int(data_dict[tf_name2])-4] = 1
                        adjacency_matrix[int(data_dict[tf_name2])-4,int(data_dict[tf_name1])-4] = 1

                if tf_root2 in interactions_dict:
                    if tf_root1 in interactions_dict[tf_root2]:
                        adjacency_matrix[int(data_dict[tf_name1])-4,int(data_dict[tf_name2])-4] = 1
                        adjacency_matrix[int(data_dict[tf_name2])-4,int(data_dict[tf_name1])-4] = 1


    return adjacency_matrix

def summarize_data(data):
    num_train = len(data['train']['tgt'])
    num_valid = len(data['valid']['tgt'])
    num_test = len(data['test']['tgt'])

    print('Num Train: '+str(num_train))
    print('Num Valid: '+str(num_valid))
    print('Num Test: '+str(num_test))

    # unconditional_probs = torch.zeros(len(data['dict']['tgt']),len(data['dict']['tgt']))
    train_label_vals = torch.zeros(len(data['train']['tgt']),len(data['dict']['tgt']))
    for i in range(len(data['train']['tgt'])):
        indices = torch.from_numpy(np.array(data['train']['tgt'][i]))
        x = torch.zeros(len(data['dict']['tgt']))
        x.index_fill_(0, indices, 1)
        train_label_vals[i] = x


    train_label_vals = train_label_vals[:,4:]

    pearson_matrix = np.corrcoef(train_label_vals.transpose(0,1).cpu().numpy())

    valid_label_vals = torch.zeros(len(data['valid']['tgt']),len(data['dict']['tgt']))
    for i in range(len(data['valid']['tgt'])):
        indices = torch.from_numpy(np.array(data['valid']['tgt'][i]))
        x = torch.zeros(len(data['dict']['tgt']))
        x.index_fill_(0, indices, 1)
        valid_label_vals[i] = x
    valid_label_vals = valid_label_vals[:,4:]

    train_valid_labels = torch.cat((train_label_vals,valid_label_vals),0)

    mean_pos_labels = torch.mean(train_valid_labels.sum(1))
    median_pos_labels = torch.median(train_valid_labels.sum(1))
    max_pos_labels = torch.max(train_valid_labels.sum(1))

    print('Mean Labels Per Sample: '+str(mean_pos_labels))
    print('Median Labels Per Sample: '+str(median_pos_labels))
    print('Max Labels Per Sample: '+str(max_pos_labels))

    mean_samples_per_label = torch.mean(train_valid_labels.sum(0))
    median_samples_per_label = torch.median(train_valid_labels.sum(0))
    max_samples_per_label = torch.max(train_valid_labels.sum(0))

    print('Mean Samples Per Label: '+str(mean_samples_per_label))
    print('Median Samples Per Label: '+str(median_samples_per_label))
    print('Max Samples Per Label: '+str(max_samples_per_label))


def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).cuda()
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.contiguous().view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)


def differentiable_f1(pred,target):
    num = 2*torch.min(pred,target).sum(1)
    den = torch.min(pred,target).sum(1) + torch.max(pred,target).sum(1)
    f1 = num/den
    return f1


def get_gold_binary_full(gold,tgt_vocab_size):
    gold_binary = torch.zeros(gold.size(0),tgt_vocab_size+4)
    for i,row in enumerate(gold_binary):
        indices = gold[i]
        indices = indices[indices > 0]
        gold_binary[i].index_fill_(0, indices, 1)
    gold_binary = gold_binary
    return gold_binary

def get_gold_binary(gold,tgt_vocab_size):
    gold_binary = torch.zeros(gold.size(0),tgt_vocab_size+4)
    for i,row in enumerate(gold_binary):
        indices = gold[i]
        indices = indices[indices > 0]
        indices = indices[0:-1]
        gold_binary[i].index_fill_(0, indices, 1)
    gold_binary = gold_binary[:,4:]
    return gold_binary


def load_embeddings(transformer,emb_path):
    print('Loading Embeddings')
    pretrained_embeddings = torch.load(emb_path)
    for word,word_idx in pretrained_embeddings['vocab'].items():
        if word in data['dict']['src']:
             data_idx = data['dict']['src'][word]
             transformer.encoder.src_word_emb.weight[data_idx].data = pretrained_embeddings['Wemb'][word_idx]
    return transformer

def save_model(opt,epoch_i,model,valid_accu,valid_accus):
    model_state_dict = model.state_dict()
    checkpoint = {'model': model_state_dict,'settings': opt,'epoch': epoch_i}
    if opt.save_mode == 'all':
        model_name = opt.model_name + '/accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
        torch.save(checkpoint, model_name)
    elif opt.save_mode == 'best':
        model_name = opt.model_name + '/model.chkpt'
        try:
            if valid_accu >= max(valid_accus):
                torch.save(checkpoint, model_name)
                print('[Info] The checkpoint file has been updated.')
        except:
            pass


