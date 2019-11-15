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
import collections
import Constants as Constants
from pdb import set_trace as stop

hg_file = Constants.hg_file
data_root = Constants.data_root
cell_line=Constants.cell_line
chroms=Constants.chroms
window_length = Constants.window_length
stride_length=Constants.stride_length

valid_chroms = ['chr3', 'chr12', 'chr17']
test_chroms = ['chr1', 'chr8', 'chr21']

alphabet = ['A','C','G','T','N']

cell_root = data_root+cell_line
processed_root=data_root+cell_line+'/processed_data/'

def create():
	SEQ_list_train = []
	SEQ_dict_train = {}

	SEQ_list_valid= []
	SEQ_dict_valid = {}

	SEQ_list_test = []
	SEQ_dict_test = {}

	with open(os.path.join(cell_type,'processed_data','all_peaks_windows.bed')) as csvfile:
		csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos','name','score','strand','signalValue','pvalue','qValue','peak'])

		k = 1
		for line in csv_reader:
			print(k)
			k+=1
			chrom = line['chrom']
			start = line['start_pos']
			tf_name = line['name']
			pvalue = line['pvalue']

			if chrom in valid_chroms:
				if start in SEQ_dict_valid:
					SEQ_dict_valid[start][tf_name] = pvalue
				else:
					SEQ_dict_valid[start] = {}
					SEQ_dict_valid[start][tf_name] = pvalue
					SEQ_list_valid.append(start)
			elif chrom in test_chroms:
				if start in SEQ_dict_test:
					SEQ_dict_test[start][tf_name] = pvalue
				else:
					SEQ_dict_test[start] = {}
					SEQ_dict_test[start][tf_name] = pvalue
					SEQ_list_test.append(start)
			else:
				if start in SEQ_dict_train:
					SEQ_dict_train[start][tf_name] = pvalue
				else:
					SEQ_dict_train[start] = {}
					SEQ_dict_train[start][tf_name] = pvalue
					SEQ_list_train.append(start)

				
			# try:
			# 	if not SEQs[-1] == start:
			# 		SEQs.append({})
			# 	else:
			# 		SEQs[-1][tf_name] = pvalue
			# except:
			# 	print('!')
			# 	SEQs.append({})


	torch.save(SEQ_list_train,'SEQ_list_train.pt')
	torch.save(SEQ_dict_train,'SEQ_dict_train.pt')

	torch.save(SEQ_list_test,'SEQ_list_test.pt')
	torch.save(SEQ_dict_test,'SEQ_dict_test.pt')

	torch.save(SEQ_list_valid,'SEQ_list_valid.pt')
	torch.save(SEQ_dict_valid,'SEQ_dict_valid.pt')



def load():

	TFs = []
	with open('sorted_TFs_'+cell_type+'.csv','r') as f:
		for line in f:
			TF_experiment = line.strip()
			# TFs[TF_experiment] = {}
			TFs.append(TF_experiment)

	SEQ_list_valid = torch.load('SEQ_list_valid.pt')
	SEQ_dict_valid= torch.load('SEQ_dict_valid.pt')

	SEQ_list_test = torch.load('SEQ_list_test.pt')
	SEQ_dict_test = torch.load('SEQ_dict_test.pt')

	SEQ_list_train = torch.load('SEQ_list_train.pt')
	SEQ_dict_train = torch.load('SEQ_dict_train.pt')


	file = open('graphical_model_data/train.csv','w')
	for TF in TFs:
		file.write(TF+',')
	file.write('\n')
	for window in SEQ_list_train:
		# file.write(window+',')
		for TF in TFs:
			if TF in SEQ_dict_train[window]:
				file.write(SEQ_dict_train[window][TF]+',')
			else:
				file.write('0,')
		file.write('\n')
	file.close()

	file = open('graphical_model_data/test.csv','w')
	for TF in TFs:
		file.write(TF+',')
	file.write('\n')
	for window in SEQ_list_test:
		# file.write(window+',')
		for TF in TFs:
			if TF in SEQ_dict_test[window]:
				file.write(SEQ_dict_test[window][TF]+',')
			else:
				file.write('0,')
		file.write('\n')
	file.close()

	file = open('graphical_model_data/valid.csv','w')
	for TF in TFs:
		file.write(TF+',')
	file.write('\n')
	for window in SEQ_list_valid:
		# file.write(window+',')
		for TF in TFs:
			if TF in SEQ_dict_valid[window]:
				file.write(SEQ_dict_valid[window][TF]+',')
			else:
				file.write('0,')
		file.write('\n')
	file.close()


# create()
load()