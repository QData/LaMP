from pdb import set_trace as stop
from torch import save
import numpy as np
import random

with open('yeast-test.arff') as f:
    lines = f.readlines()

test_inputs = open('test_inputs.txt','w')
test_labels = open('test_labels.txt','w')

Flag = False

word_list = []
label_list = []
attr_list = []

for line_idx,line in enumerate(lines):
	line = line.strip()

	if line_idx < 2:
		pass
	elif line_idx < 105:
		split_line = line.split(' ')
		word_list.append(split_line[1])
		attr_list.append(split_line[1])
	elif line_idx < 119:
		split_line = line.split(' ')
		label_list.append(split_line[1])
		attr_list.append(split_line[1])
	elif line_idx < 121:
		pass
	else:
		split_line = line.split(',')
		features = split_line[0:103]
		labels = split_line[103:]

		for feature in features:
			test_inputs.write(feature+' ')
		for idx, label in enumerate(labels):
			if label == '1':
				test_labels.write(label_list[idx]+' ')
		
		test_inputs.write('\n')
		test_labels.write('\n')

test_inputs.close()
test_inputs.close()





with open('yeast-train.arff') as f:
	lines = f.readlines()

train_inputs = open('train_inputs.txt','w')
train_labels = open('train_labels.txt','w')

valid_inputs = open('valid_inputs.txt','w')
valid_labels = open('valid_labels.txt','w')

word_list = []
label_list = []
attr_list = []

for line_idx,line in enumerate(lines):
	line = line.strip()

	if line_idx < 2:
		pass
	elif line_idx < 105:
		split_line = line.split(' ')
		word_list.append(split_line[1])
		attr_list.append(split_line[1])
	elif line_idx < 119:
		split_line = line.split(' ')
		label_list.append(split_line[1])
		attr_list.append(split_line[1])
	elif line_idx < 121:
		if line == '@data':
			num_valid_samples = int( ((len(lines)+1)-(line_idx+1))/5 )
			valid_indices = random.sample(range(line_idx+1, len(lines)),  num_valid_samples)
	else:
		split_line = line.split(',')
		features = split_line[0:103]
		labels = split_line[103:]

		if line_idx in valid_indices:	
			for feature in features:
				valid_inputs.write(feature+' ')
			for idx, label in enumerate(labels):
				if label == '1':
					valid_labels.write(label_list[idx]+' ')
			valid_inputs.write('\n')
			valid_labels.write('\n')
		else:

			for feature in features:
				train_inputs.write(feature+' ')
			for idx, label in enumerate(labels):
				if label == '1':
					train_labels.write(label_list[idx]+' ')
		
			train_inputs.write('\n')
			train_labels.write('\n')
			
		
		


train_inputs.close()
train_labels.close()
valid_inputs.close()
valid_labels.close()



