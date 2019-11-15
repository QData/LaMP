#!/usr/bin/env python

import math
import csv
import glob, os
import sys
import string
import pyBigWig
import collections
import time
import pickle
import progressbar
import os.path
import argparse


parser = argparse.ArgumentParser(description='Fetching HM Data')
parser.add_argument('--cell_line', type=str, default='GM12878', help='GM12878 or Tcell')
parser.add_argument('--HM', type=str, default='', help='Specify HM to process. Leave blank to loop through all')
args = parser.parse_args()

cell_line = args.cell_line



SEQ_length = 200
SEQ_length_half = math.floor(SEQ_length/2)

HM_length = 2000
HM_resolution = 20
HM_bins = int(math.floor(HM_length/HM_resolution))
HM_length_half = math.floor(HM_length/2)-SEQ_length_half




root_dir = '/af11/jjl5sw/chongzhi_datasets/'+cell_line+'/'


print('======> Finding Chrom Lengths')
chrom_lengths = {}
with open('/bigtemp/jjl5sw/hg38/hg38.chrom_sizes') as csvfile:
	csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom_name','length'])
	for csv_row in csv_reader:
		chrom_lengths[csv_row['chrom_name']] = csv_row['length']



num_total_samples = sum(1 for line in open(root_dir+'windows.bed'))


def gen_histone(bw_file_name,bed_file_name):

		bw_file =  pyBigWig.open(bw_file_name)
		hm_bed_file = open(bed_file_name,'w')


		with open(root_dir+'/windows.bed') as bed_file:
			bed_windows = csv.DictReader(bed_file,delimiter='\t',fieldnames=['chrom','start_pos','end_pos'])
			with progressbar.ProgressBar(max_value=num_total_samples) as bar:
				k = 0
				for window in bed_windows:
					bar.update(k); k+=1

					chrom = window['chrom']
					start_pos = int(window['start_pos'])
					end_pos = int(window['end_pos'])

					HM_start = max(0,int(start_pos-HM_length_half))
					HM_end = min(chrom_lengths[chrom],int(end_pos+HM_length_half))

					values = bw_file.stats(chrom,HM_start,HM_end,nBins=HM_bins)


					hm_bed_file.write(chrom+'\t'+str(start_pos)+'\t'+str(end_pos)+'\t')

					hm_bed_file.write(str(values[0]))
					for i in range(1,len(values)):
						value = values[i]
						hm_bed_file.write(','+str(value))

					hm_bed_file.write('\n')


			hm_bed_file.close()



def main():
	for bw_file_name in glob.glob(root_dir+'HM/'+args.HM+'*.bw'):
		bed_file_name = bw_file_name.replace('/HM/','/Data/HM/').replace('bw','txt')
		if not os.path.isfile(bed_file_name):
			print(bed_file_name)
			gen_histone(bw_file_name,bed_file_name)
		else:
			print(bed_file_name+' exists...skipping')


if __name__ == "__main__":
	main()



# try:
# 	average = [float(sum(y)) / len(y) for y in zip(*values)]
# 	for value in average:
# 		hm_bed_file.write(str(value)+',')
# except:
# 	for v in range(0,100):
# 		hm_bed_file.write('0.0,')
