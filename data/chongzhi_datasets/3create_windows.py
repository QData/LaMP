#!/usr/bin/env python

# Finds TF bindings for the 200-length windows defined in /bigtemp/jjl5sw/chongzhi_datasets/hg38_200.bed


import csv
import os, glob
import Constants as Constants


hg_file = Constants.hg_file
data_root = Constants.data_root
cell_line=Constants.cell_line
chroms=Constants.chroms
window_length = Constants.window_length
stride_length=Constants.stride_length


root=data_root+cell_line+'/processed_data/'

bed_windows_file = data_root+'hg38_windows.bed'

file=root+'all_peaks.bed'
file_2=root+'all_peaks_windows.tmp'
file_3=root+'all_peaks_windows.bed'


print('Input1: '+bed_windows_file)
print('Input2: '+file)
print('Output: '+file_3)


# Any intersection over 50% of windows surrounding peak
sys_command='bedtools intersect -wa -wb -f 0.5 -a '+bed_windows_file+' -b  '+file+' > '+file_2
os.system(sys_command)


#awk '{print $1"\t"$2"\t"$3"\t"$7"\t"$8"\t"$9"\t"$10"\t"$11"\t"$12"\t"$13}' $file_2 >> $file_3
sys_command2 = '''awk \'{print $1\"\\t\"$2\"\\t\"$3\"\\t\"$7\"\\t\"$8\"\\t\"$9\"\\t\"$10\"\\t\"$11\"\\t\"$12\"\\t\"$13}\' '''+file_2+' > '+file_3
os.system(sys_command2)

sys_command3='rm '+file_2
os.system(sys_command3)
