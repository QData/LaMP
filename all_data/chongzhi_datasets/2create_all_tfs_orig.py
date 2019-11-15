#!/usr/bin/env python

# creates a single file for all TFs. 'name' column in the narrowPeak bed file
# is the TF ID

import csv
import os, glob
import Constants as Constants
# import sys
# from random import randint, choice
# import collections



data_root = Constants.data_root
cell_line=Constants.cell_line
chroms=Constants.chroms
window_length = Constants.window_length


cell_line_dir = data_root+cell_line

window_length_half = int(float(window_length)/2)



input_file_name = os.path.join(cell_line_dir,'original_data/TF/*.bed')
output_file_name = os.path.join(cell_line_dir,'processed_data/all_peaks.bed')
print('Input: '+input_file_name)
print('Output: '+output_file_name)


output_file =  open(output_file_name, 'w')


all_seq_lengths = []

for file_path in glob.glob(input_file_name):
	file_dir, file_name = os.path.split(file_path)
	TF_id = file_name.split('.')[0]
	print(TF_id)
	with open(file_path) as csvfile:
		csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos','name','score','strand','signalValue','pvalue','qValue','peak'])
		for sample in csv_reader:
			chrom = sample['chrom']
			if chrom in chroms:
				start_pos = sample['start_pos']
				end_pos = sample['end_pos']
				score = sample['score']
				strand = sample['strand']
				signalValue = sample['signalValue']
				pvalue = sample['pvalue']
				qValue = sample['qValue']
				peak = sample['peak']

				

				## METHOD A ##
				# peak_pos = int(start_pos)+int(peak)
				# output_file.write(chrom+'\t'+str(peak_pos-window_length_half)+'\t'+str(peak_pos+window_length_half)+'\t'+TF_id+'\t'+score+'\t'+strand+'\t'+signalValue+'\t'+pvalue+'\t'+qValue+'\t'+str(window_length_half)+'\n')

				## METHOD B ##
				all_seq_lengths.append(int(end_pos)-int(start_pos))
				output_file.write(chrom+'\t'+str(start_pos)+'\t'+str(end_pos)+'\t'+TF_id+'\t'+score+'\t'+strand+'\t'+signalValue+'\t'+pvalue+'\t'+qValue+'\t'+str(peak)+'\n')


output_file.close()

import numpy as np
all_seq_lengths = np.array(all_seq_lengths)
print(np.max(all_seq_lengths))
print(np.mean(all_seq_lengths))