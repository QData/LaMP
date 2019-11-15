import torch
import numpy as np
from scipy.sparse import coo_matrix,save_npz
data =torch.load('train_valid_test.pt')

num_labels = len(data['dict']['tgt'])-4
num_samples = len(data['train']['tgt'])
adj_mat = torch.zeros(num_labels,num_labels)

num_pos_indices = 1000

for i,sample in enumerate(data['train']['tgt']):
    print("{}/{}".format(i, num_samples))
    pos_labels = sample[1:-1]
    for idx1 in range(len(pos_labels)-1):
        for idx2 in range(idx1+1,len(pos_labels)):
            adj_mat[pos_labels[idx1]-4,pos_labels[idx2]-4] += 1
            adj_mat[pos_labels[idx2]-4,pos_labels[idx1]-4] += 1

sorted_vals, indices = torch.sort(adj_mat.view(-1), 0,descending=True)

max_val = sorted_vals[num_pos_indices]

adj_mat[adj_mat<max_val] = 0
adj_mat[adj_mat>=max_val] = 1

adj_coo = coo_matrix(adj_mat.numpy())

save_npz('adj_matrix.npz', adj_coo)


# indices = torch.nonzero(adj_mat).t()
# values = adj_mat[indices[0], indices[1]]
# adj_sparse = torch.sparse.FloatTensor(indices, values, adj_mat.size())


# torch.save(adj_sparse,'adj_matrix.pt')
# np.percentile(adj_mat.numpy(), 99.999)