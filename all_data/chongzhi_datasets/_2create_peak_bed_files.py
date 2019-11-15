#!/usr/bin/env python
# don't really need this anymore

# Takes about 1 minute for 179 TF files

import csv
import math
import os, glob
import sys
from random import randint, choice
import collections
import pickle


window_length = 200
window_length_half = int(float(window_length)/2)


try:
	cell_line = sys.argv[1]
except:
	cell_line = 'GM12878'
	# cell_line = 'Tcell'

orig_file_dir = os.path.join(cell_line,'original_data','TF')
output_file_dir = os.path.join(cell_line,'processed_data','TF')

for file_path in glob.glob(os.path.join(orig_file_dir,'*.bed')):
	file_dir, file_name = os.path.split(file_path)
	new_file_path = os.path.join(output_file_dir,file_name)
	print file_path
	print new_file_path

	output_file =  open(new_file_path, 'w')
	with open(file_path) as csvfile:
		csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos','name','score','strand','signalValue','pvalue','qValue','peak'])
		for sample in csv_reader:
			chrom = sample['chrom']
			start_pos = sample['start_pos']
			end_pos = sample['end_pos']
			name = sample['name']
			score = sample['score']
			strand = sample['strand']
			signalValue = sample['signalValue']
			pvalue = sample['pvalue']
			qValue = sample['qValue']
			peak = sample['peak']


			peak_pos = int(start_pos)+int(peak)

			# output_file.write(chrom+'\t'+str(start_pos)+'\t'+str(end_pos)+'\t'+name+'\t'+score+'\t'+strand+'\t'+signalValue+'\t'+pvalue+'\t'+qValue+'\t'+peak+'\n')
	        output_file.write(chrom+'\t'+str(peak_pos-window_length_half)+'\t'+str(peak_pos+window_length_half)+'\t'+name+'\t'+score+'\t'+strand+'\t'+signalValue+'\t'+pvalue+'\t'+qValue+'\t'+str(window_length_half)+'\n')
	output_file.close()
