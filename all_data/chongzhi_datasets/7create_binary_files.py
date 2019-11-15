#!/usr/bin/env python

import os
import torch
import sys
import csv
import glob
import pickle
from multiprocessing import Process
import random
import os.path
import sys
import time
import Constants as Constants
from pdb import set_trace as stop

# from numpy import genfromtxt
# my_data = genfromtxt('my_file.csv', delimiter=',')



hg_file = Constants.hg_file
data_root = Constants.data_root
cell_line=Constants.cell_line
chroms=Constants.chroms
window_length = Constants.window_length
stride_length=Constants.stride_length


cell_root = data_root+cell_line
processed_root=data_root+cell_line+'/processed_data/'

# test_chroms = ['chr3','chr10','chr20']
# valid_chroms = ['chr4','chr11','chr19']

valid_chroms = ['chr3', 'chr12', 'chr17']
test_chroms = ['chr1', 'chr8', 'chr21']


alphabet = ['A','C','G','T','N']


t1 = time.time()
with open(os.path.join(processed_root,'windows_and_pvalues.txt'), "r") as csvfile:
	headers = next(csvfile).split()
	TF_names = headers[4:]
	TF_lists_dict = {}
	for TF in TF_names:
		TF_lists_dict[TF] = []
	SEQ_list = []
	csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=headers)
	k = 0

	train_idxs = []
	valid_idxs = []
	test_idxs = []

	for line in csv_reader:
		chrom = line['chrom']
		start = line['start_pos']

		if chrom in test_chroms:
			test_idxs.append(k)
		elif chrom in valid_chroms:
			valid_idxs.append(k)
		else:
			train_idxs.append(k)

		seq = line['SEQ']
		for idx, char in enumerate(alphabet):
			seq = seq.replace(char,str(idx))
		seq = map(int,list(seq))
		SEQ_list.append(seq)
		for TF in TF_names:
			pvalue = float(line[TF])
			TF_lists_dict[TF].append(pvalue)
		if k%100000 == 0:
			print(float(k)/2297493)
		# if (float(k)/2297493) > 0.3:
		# 	break
		k+=1

SEQ_tensor = torch.Tensor(SEQ_list)
num_samples = SEQ_tensor.size(0)


train_idxs = torch.LongTensor(train_idxs)
valid_idxs = torch.LongTensor(valid_idxs)
test_idxs = torch.LongTensor(test_idxs)


print(train_idxs)

train_shuff_indices = torch.randperm(train_idxs.size(0))
valid_shuff_indices = torch.randperm(valid_idxs.size(0))
test_shuff_indices = torch.randperm(test_idxs.size(0))


print('Saving SEQs')
SEQ_tensor_train = torch.index_select(SEQ_tensor, dim=0, index=train_shuff_indices)
SEQ_tensor_valid = torch.index_select(SEQ_tensor, dim=0, index=valid_shuff_indices)
SEQ_tensor_test = torch.index_select(SEQ_tensor, dim=0, index=test_shuff_indices) 

torch.save(SEQ_tensor_train,os.path.join(cell_root,'train','SEQs.seq'))
torch.save(SEQ_tensor_valid,os.path.join(cell_root,'valid','SEQs.seq'))
torch.save(SEQ_tensor_test,os.path.join(cell_root,'test','SEQs.seq'))

print('Saving TFs')
for TF in TF_names:
	print(TF)
	TF_tensor = torch.Tensor(TF_lists_dict[TF])

	TF_tensor_train = torch.index_select(TF_tensor, dim=0, index=train_shuff_indices)# TF_tensor.index(train_idxs)[train_shuff_indices]
	TF_tensor_valid= torch.index_select(TF_tensor, dim=0, index=valid_shuff_indices) #TF_tensor.index(valid_idxs)[valid_shuff_indices]
	TF_tensor_test = torch.index_select(TF_tensor, dim=0, index=test_shuff_indices)# TF_tensor.index(test_idxs)[test_shuff_indices]

	torch.save(TF_tensor_train,os.path.join(cell_root,'train',TF+'.tf'))
	torch.save(TF_tensor_valid,os.path.join(cell_root,'valid',TF+'.tf'))
	torch.save(TF_tensor_test,os.path.join(cell_root,'test',TF+'.tf'))

t2 = time.time()
print(str(t2-t1)+' seconds')



### OLD METHOD: randomly sample train and valid samples
# t1 = time.time()
# with open(os.path.join(processed_root,'windows_and_pvalues.txt')) as csvfile:
# 	headers = csvfile.next().split()
# 	TF_names = headers[4:]
# 	TF_lists_dict = {}
# 	for TF in TF_names:
# 		TF_lists_dict[TF] = []
# 	SEQ_list = []
# 	csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=headers)
# 	k = 0
# 	for line in csv_reader:
# 		k+=1
# 		chrom = line['chrom']
# 		start = line['start_pos']
# 		# end = line['end_pos']
# 		seq = line['SEQ']
# 		for idx, char in enumerate(alphabet):
# 			seq = seq.replace(char,str(idx))
# 		seq = map(int,list(seq))
# 		SEQ_list.append(seq)
# 		for TF in TF_names:
# 			pvalue = float(line[TF])
# 			TF_lists_dict[TF].append(pvalue)
# 		if k%100000 == 0:
# 			print(float(k)/2297493)
#
#
# SEQ_tensor = torch.Tensor(SEQ_list)
# num_samples = SEQ_tensor.size(0)
# train_start = 0
# train_end = int(num_samples*0.7)
# valid_start = train_end
# valid_end = valid_start+int(num_samples*0.15)
# test_start = valid_end
# test_end = num_samples
# shuff_indices = torch.randperm(num_samples)
#
# print('Saving SEQs')
# SEQ_tensor = SEQ_tensor[shuff_indices]
# torch.save(SEQ_tensor[train_start:train_end],os.path.join(cell_type,'train','SEQs.seq'))
# torch.save(SEQ_tensor[valid_start:valid_end],os.path.join(cell_type,'valid','SEQs.seq'))
# torch.save(SEQ_tensor[test_start:test_end],os.path.join(cell_type,'test','SEQs.seq'))
#
# print('Saving TFs')
# for TF in TF_names:
# 	print(TF)
# 	TF_tensor = torch.Tensor(TF_lists_dict[TF])
# 	TF_tensor = TF_tensor[shuff_indices]
# 	torch.save(TF_tensor[train_start:train_end],os.path.join(cell_type,'train',TF+'.tf'))
# 	torch.save(TF_tensor[valid_start:valid_end],os.path.join(cell_type,'valid',TF+'.tf'))
# 	torch.save(TF_tensor[test_start:test_end],os.path.join(cell_type,'test',TF+'.tf'))
#
# t2 = time.time()
# print(str(t2-t1)+' seconds')
