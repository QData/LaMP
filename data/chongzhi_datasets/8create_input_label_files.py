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
stride_length=Constants.stride_length


cell_root = data_root+cell_line
processed_root=data_root+cell_line+'/processed_data/'

# test_chroms = ['chr3','chr10','chr20']
# valid_chroms = ['chr4','chr11','chr19']

valid_chroms = ['chr3', 'chr12', 'chr17']
test_chroms = ['chr1', 'chr8', 'chr21']


alphabet = ['A','C','G','T','N']


train_inputs = open('train_inputs.txt','w')
train_labels= open('train_labels.txt','w')

valid_inputs = open('valid_inputs.txt','w')
valid_labels= open('valid_labels.txt','w')

test_inputs = open('test_inputs.txt','w')
test_labels= open('test_labels.txt','w')


input_file = os.path.join(processed_root,'windows_and_pvalues.txt')
print(input_file)

t1 = time.time()
with open(input_file, "r") as csvfile:
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
		seq = line['SEQ']
		split_seq = [char for char in seq]
		split_seq = " ".join(map(str, split_seq))

		if chrom in test_chroms:
			test_idxs.append(k)
			test_inputs.write(split_seq+'\n')
			for TF in TF_names:
				pvalue = float(line[TF])
				if pvalue > 0.0:
					test_labels.write(TF+' ')
			test_labels.write('\n')
		elif chrom in valid_chroms:
			valid_idxs.append(k)
			valid_inputs.write(split_seq+'\n')
			for TF in TF_names:
				pvalue = float(line[TF])
				if pvalue > 0.0:
					valid_labels.write(TF+' ')
			valid_labels.write('\n')
		else:
			train_idxs.append(k)
			
			train_inputs.write(split_seq+'\n')
			for TF in TF_names:
				pvalue = float(line[TF])
				if pvalue > 0.0:
					train_labels.write(TF+' ')
			train_labels.write('\n')

		if k%100000 == 0:
			print(float(k)/1007671)
		# if (float(k)/2297493) > 0.05:
		# 	break
		k+=1


train_inputs.close()
train_labels.close()

valid_inputs.close()
valid_labels.close()

test_inputs.close()
test_labels.close()



