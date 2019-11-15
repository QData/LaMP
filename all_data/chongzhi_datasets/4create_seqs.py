#!/usr/bin/env python

import os
import sys
import Constants as Constants
import csv

hg_file = Constants.hg_file
data_root = Constants.data_root
cell_line=Constants.cell_line
chroms=Constants.chroms
window_length = Constants.window_length
stride_length=Constants.stride_length


root=data_root+cell_line+'/processed_data/'

windows_file_name = root+'all_peaks_windows.bed'
extended_windows_file_name = root+'all_peaks_windows_extended.bed'
fasta_file_name= root+'all_seqs_windows_extended.fa'

print(windows_file_name)
print(extended_windows_file_name)
print(fasta_file_name)



def create_extended_windows():
	with open(extended_windows_file_name,'w') as output_file:
		with open(windows_file_name) as csvfile:
			csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos','name','score','strand','signalValue','pvalue','qValue','peak'])
			for sample in csv_reader:
				chrom = sample['chrom']
				start_pos = sample['start_pos']
				end_pos = sample['end_pos']
				score = sample['score']
				strand = sample['strand']
				signalValue = sample['signalValue']
				pvalue = sample['pvalue']
				qValue = sample['qValue']
				peak = sample['peak']
				TF_id = sample['name']

				peak_pos = int(start_pos)+int(peak)

				# 1000 length around the center of each window
				start_pos = int(start_pos)-400
				end_pos = int(end_pos)+400

				output_file.write(chrom+'\t'+str(start_pos)+'\t'+str(end_pos)+'\t'+TF_id+'\t'+score+'\t'+strand+'\t'+signalValue+'\t'+pvalue+'\t'+qValue+'\t'+str(peak)+'\n')


def create_fa():
	# cmd = 'bedtools getfasta -fi /bigtemp/jjl5sw/hg38/hg38.fa -bed '+windows_file_name+' -fo '+fasta_file_name

	cmd = 'bedtools getfasta -fi /bigtemp/jjl5sw/hg38/hg38.fa -bed '+extended_windows_file_name+' -fo '+fasta_file_name
	os.system(cmd)


def create_bed_seqs():
	bed_seqs_file_name = fasta_file_name.replace('.fa','.bed')
	bed_seqs_file = open(bed_seqs_file_name,'w')

	f = open(fasta_file_name,'r')

	for line in f:
		if '>' in line:
			chrom = line.split('>')[1].split(':')[0]
			start_pos = int(line.split(':')[1].split('-')[0])
			end_pos = int(line.split(':')[1].split('-')[1])
		else:
			bed_seqs_file.write(chrom+'\t'+str(start_pos)+'\t'+str(end_pos)+'\t'+line.upper())

	f.close()
	bed_seqs_file.close()

	os.system('rm '+fasta_file_name)


create_extended_windows()
create_fa()
create_bed_seqs()
