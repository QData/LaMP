from scipy.io import loadmat
import numpy as np
import torch

x = loadmat('ALL_inputs.mat')['ALL_inputs']
y = loadmat('Labels.mat')['Labels']

x = x.todense()
x = x[1:,1:]

x = torch.from_numpy(x)

y = torch.from_numpy(y)

y[y != y] = 0

y_nonzero = (y != 0.0).sum(dim=0)

y_feature_indices = []
for idx,value in enumerate(y_nonzero):
	if value.item() >= 15:
		y_feature_indices.append(idx)
y_feature_indices = torch.Tensor(y_feature_indices).long()



x_nonzero = (x != 0.0).sum(dim=0)


x_feature_indices = []
for idx,value in enumerate(x_nonzero):
	if value.item() >= 2:
		x_feature_indices.append(idx)
x_feature_indices = torch.Tensor(x_feature_indices).long()


x = torch.index_select(x, 1, x_feature_indices)

inputs_file = open('all_inputs.txt','w+')
for idx,sample in enumerate(x):
	print(idx)
	for value in sample[:-1]:
		inputs_file.write(str(value.item())+' ')
	inputs_file.write(str(sample[-1].item())+'\n')



labels_file = open('all_labels.txt','w+')
for idx,sample in enumerate(y):
	print(idx)
	nonzero = sample.nonzero()
	if len(nonzero) > 0:
		for value in nonzero[:-1]:
			labels_file.write(str(int(value.item()))+' ')
		labels_file.write(str(int(nonzero[-1].item())))
	labels_file.write('\n')

