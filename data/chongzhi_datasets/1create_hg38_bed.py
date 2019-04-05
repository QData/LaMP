# Generates 200 length windows with stride of 50 in hg38

import csv
import math
import Constants as Constants



hg_file = Constants.hg_file
data_root = Constants.data_root
cell_line=Constants.cell_line
chroms=Constants.chroms
window_length = Constants.window_length
stride_length=Constants.stride_length


input_file_name = hg_file
output_file_name = data_root+'hg38_windows.bed'
print('Input: '+input_file_name)
print('Output: '+output_file_name)

output_file =  open(output_file_name, 'w')




lengths = {}
with open(input_file_name) as csvfile:
	csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom_name','length'])
	for csv_row in csv_reader:
		lengths[csv_row['chrom_name']] = csv_row['length']


for chrom in chroms:
	chrom_length = lengths[chrom]
	prev_i = 0
	for i in range(0,int(chrom_length),stride_length):
		output_file.write(chrom+'\t'+str(prev_i)+'\t'+str(prev_i+window_length)+'\n')
		prev_i = i+stride_length
