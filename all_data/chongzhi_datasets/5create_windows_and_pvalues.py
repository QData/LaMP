#!/usr/bin/env python

################################################################################
# Finds all the TFBS windows from the *_windows.bed files and creates the
# negative binding sites from them. Writes individual or single TF binding files

# Inputs: root+'chongzhi_datasets/'+cell_line+'/TF/*_windows.bed
# Outputs: root+'chongzhi_datasets/'+cell_line+'/Data/TF/'+str(TF_id)+'.txt' TF binding files
#################################################################################

import csv
import math
import os, glob
import sys
from random import randint, choice
import collections
import pickle
import Constants as Constants
from pdb import set_trace as stop

hg_file = Constants.hg_file
data_root = Constants.data_root
cell_line=Constants.cell_line
chroms=Constants.chroms
extended_window_length = Constants.extended_window_length
stride_length=Constants.stride_length

processed_root=data_root+cell_line+'/processed_data/'

#global_dict[chrom][start_pos]['TFs'] = {}
#global_dict[chrom][start_pos]['sequence'] = str
#global_dict[chrom][start_pos]['HMs'] = {}

################################################################################
######### Create dictionary of TFs binding to each 200 length window ###########
################################################################################
print('=======> Creating TF Binding Dictionary')
global_dict = {} #Format: global_dict[chrom][start_pos]['TFs'][TF_id] = pvalue
for chrom in chroms:
	global_dict[chrom]={}


all_peaks_file_name = processed_root+'all_peaks_windows_extended.bed'
all_seqs_file_name = processed_root+'all_seqs_windows_extended.bed'
output_file_name = processed_root+'windows_and_pvalues.txt'
print(all_peaks_file_name)
print(output_file_name)

TF_ids = []
with open(all_peaks_file_name) as csvfile:
	csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos','TF_id','score','strand','signalValue','pvalue','qValue','peak'])
	for csv_row in csv_reader:
		TF_id = str(csv_row['TF_id'])
		chrom = str(csv_row['chrom'])
		start_pos = int(csv_row['start_pos'])
		pvalue = float(csv_row['pvalue'])

		if not TF_id in TF_ids:
			TF_ids.append(TF_id)

		if not start_pos in global_dict[chrom]:
			global_dict[chrom][start_pos] = {}
			global_dict[chrom][start_pos]['TFs'] = {}

		global_dict[chrom][start_pos]['TFs'][TF_id] = pvalue


# sort each chromosome by start position
for chrom in global_dict:
	global_dict[chrom] = collections.OrderedDict(sorted(global_dict[chrom].items()))


################################################################################
######################### Write TF Binding Bed File(s) #########################
################################################################################
write_single_tf_file = True
write_individual_tf_files = False

print(processed_root)

##### Write One Global TF File #####
if write_single_tf_file:
	print('=======> Writing Single TF File')

	seqs_dict = {}
	with open(all_seqs_file_name) as csvfile:
		csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos','SEQ'])
		for line in csv_reader:
			chrom = line['chrom']
			start_pos = int(line['start_pos'])
			if not chrom in seqs_dict:
				seqs_dict[chrom] = {}
			seqs_dict[chrom][start_pos] = line['SEQ']


	output_file =  open(output_file_name, 'w')
	output_file.write('chrom\tstart_pos\tend_pos\tSEQ')
	for TF_id in TF_ids:
		output_file.write('\t'+str(TF_id))
	output_file.write('\n')

	for chrom in chroms:
		for start_pos in global_dict[chrom]:
			output_file.write(str(chrom)+'\t'+str(start_pos)+'\t'+str(start_pos+extended_window_length))
			output_file.write('\t'+seqs_dict[chrom][start_pos])
			for TF_id in TF_ids:
				if TF_id in global_dict[chrom][start_pos]['TFs']:
					output_file.write('\t'+str(global_dict[chrom][start_pos]['TFs'][TF_id]))
				else:
					output_file.write('\t0')
			output_file.write('\n')
	output_file.close()


##### Write individual TF files ######
if write_individual_tf_files:
	print('=======> Writing Individual TF Files')
	for TF_id in TF_ids:
		# print(TF_id)
		output_file =  open(processed_root+'/TF/'+str(TF_id)+'.txt', 'w')
		for chrom in chroms:
			for start_pos in global_dict[chrom]:
				if TF_id in global_dict[chrom][start_pos]['TFs']:
					output_file.write(str(chrom)+'\t'+str(start_pos)+'\t'+str(start_pos+extended_window_length)+'\t'+str(global_dict[chrom][start_pos]['TFs'][TF_id])+'\n')
				else:
					output_file.write(str(chrom)+'\t'+str(start_pos)+'\t'+str(start_pos+extended_window_length)+'\t'+str(0)+'\n')




### Global Windows Bed File ####
# print('=======> Writing Global Seq Bed File')
# output_file =  open(cell_line+'/windows.bed', 'w')
# for chrom in chroms:
# 	for start_pos in global_dict[chrom]:
# 		output_file.write(str(chrom)+'\t'+str(start_pos)+'\t'+str(start_pos+window_length)+'\n')
# output_file.close()



################################################################################
#################### Create Negative Sites from Positives ######################
################################################################################
# print('=======> Creating Negative Binding Sites')
# lengths = {}
# with open('/bigtemp/jjl5sw/hg38/hg38.chrom_sizes') as csvfile:
# 	csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom_name','length'])
# 	for csv_row in csv_reader:
# 		lengths[csv_row['chrom_name']] = csv_row['length']
#
# def roundup(x):
#     return int(math.ceil(x / float(window_length))) * window_length
#
#
# #find negative sites within +/- 2000-10000 bp away from positive sites
# max_dist = 10000
# min_dist = 2000
# for chrom in global_dict:
# 	for start_pos in global_dict[chrom].keys():
# 		increment = roundup(choice([randint(-max_dist,-(min_dist)),randint((min_dist),(max_dist))]))
# 		neg_start_pos = start_pos + increment
# 		if (neg_start_pos in global_dict[chrom]) or (neg_start_pos < 0) or (neg_start_pos > lengths[chrom]):
# 			proceed = False
# 			while not proceed:
# 				proceed = True
# 				increment = roundup(choice([randint(-max_dist,-(min_dist)),randint((min_dist),(max_dist))]))
# 				neg_start_pos = start_pos + increment
# 				if (neg_start_pos in global_dict[chrom]) or (neg_start_pos < 0) or (neg_start_pos > lengths[chrom]):
# 					proceed = False
# 		global_dict[chrom][neg_start_pos] = {}
# 		global_dict[chrom][neg_start_pos]['TFs'] = {}
# 		global_dict[chrom][neg_start_pos]['TFs']['None'] = 0


################################################################################
############################## Print TF Counts #################################
################################################################################

# print('=======> Retrieving TF counts')
# positions = 0
# for chrom in chroms:
# 	for start_pos in global_dict[chrom]:
# 		positions+=1
# TF_counts = {}
# for TF_id in TF_ids:
# 	TF_counts[TF_id] = 0
# 	for chrom in chroms:
# 		for start_pos in global_dict[chrom]:
# 			chrom_pos = str(chrom)+'_'+str(start_pos)
# 			if TF_id in global_dict[chrom][start_pos]['TFs']:
# 				TF_counts[TF_id] += 1
# print('total positions,'+str(positions))
# for TF_id in TF_counts:
# 	print(str(TF_id)+','+str(TF_counts[TF_id]))
