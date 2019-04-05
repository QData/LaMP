from pdb import set_trace as stop
from torch import save
import numpy as np
import random

with open('nus-wide-full-cVLADplus-test.arff') as f:
    lines = f.readlines()

test_inputs = open('test_inputs.txt','w')
test_labels = open('test_labels.txt','w')

Flag = False
label_list = []
for line in lines:
	line = line.strip()
	split_line = line.split(' ')
	if split_line[-1] == '{0,1}':
		label_list.append(split_line[1])
	elif Flag == True:
		split_line = line.split(',')
		features = split_line[1:129]
		labels = split_line[129:]
		for feature in features:
			test_inputs.write(feature+' ')
		test_inputs.write('\n')
		for idx, label in enumerate(labels):
			if label == '1':
				test_labels.write(label_list[idx]+' ')
		test_labels.write('\n')
	elif split_line[0] == '@data':
		Flag = True


test_inputs.close()
test_inputs.close()





with open('nus-wide-full-cVLADplus-train.arff') as f:
    lines = f.readlines()

train_inputs = open('train_inputs.txt','w')
train_labels = open('train_labels.txt','w')

valid_inputs = open('valid_inputs.txt','w')
valid_labels = open('valid_labels.txt','w')

Flag = False
label_list = []
for line_idx,line in enumerate(lines):
	line = line.strip()
	split_line = line.split(' ')
	if split_line[-1] == '{0,1}':
		label_list.append(split_line[1])
	elif Flag == True:
		split_line = line.split(',')
		features = split_line[1:129]
		labels = split_line[129:]
		
		if line_idx in valid_indices:
			for feature in features:
				valid_inputs.write(feature+' ')
			valid_inputs.write('\n')
			for idx, label in enumerate(labels):
				if label == '1':
					valid_labels.write(label_list[idx]+' ')
			valid_labels.write('\n')
		else:
			for feature in features:
				train_inputs.write(feature+' ')
			train_inputs.write('\n')
			for idx, label in enumerate(labels):
				if label == '1':
					train_labels.write(label_list[idx]+' ')
			train_labels.write('\n')
			
	elif split_line[0] == '@data':
		Flag = True
		num_valid_samples = int( ((len(lines)+1)-(line_idx+1))/5 )
		valid_indices = random.sample(range(line_idx+1, len(lines)),  num_valid_samples)


train_inputs.close()
train_labels.close()
valid_inputs.close()
valid_labels.close()



