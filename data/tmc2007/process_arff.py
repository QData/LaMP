from pdb import set_trace as stop
from torch import save
import numpy as np
import random

with open('tmc2007-500-test.arff') as f:
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
	elif line_idx < 502:
		split_line = line.split(' ')
		word_list.append(split_line[1])
		attr_list.append(split_line[1])
	elif line_idx < 524:
		split_line = line.split(' ')
		label_list.append(split_line[1])
		attr_list.append(split_line[1])
	elif line_idx < 526:
		pass
	else:
		split_line = line.replace('{','').replace('}','').split(',')
		for attribute_count in split_line:
			attr_idx,count = attribute_count.split(' ')
			attr_idx,count = int(attr_idx),int(count)

			attribute = attr_list[attr_idx]

			if attribute in label_list:
				test_labels.write(attribute+' ')
			else:
				test_inputs.write(attribute+' ')
		
		test_inputs.write('\n')
		test_labels.write('\n')

test_inputs.close()
test_inputs.close()





with open('tmc2007-500-train.arff') as f:
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
	elif line_idx < 502:
		split_line = line.split(' ')
		word_list.append(split_line[1])
		attr_list.append(split_line[1])
	elif line_idx < 524:
		split_line = line.split(' ')
		label_list.append(split_line[1])
		attr_list.append(split_line[1])
	elif line_idx < 526:
		if line == '@data':
			num_valid_samples = int( ((len(lines)+1)-(line_idx+1))/6 )
			valid_indices = random.sample(range(line_idx+1, len(lines)),  num_valid_samples)
	else:
		split_line = line.replace('{','').replace('}','').split(',')
		for attribute_count in split_line:
			attr_idx,count = attribute_count.split(' ')
			attr_idx,count = int(attr_idx),int(count)

			attribute = attr_list[attr_idx]

			if attribute in label_list:
				if line_idx in valid_indices:
					valid_labels.write(attribute+' ')
				else:
					train_labels.write(attribute+' ')
			else:
				if line_idx in valid_indices:
					valid_inputs.write(attribute+' ')
				else:
					train_inputs.write(attribute+' ')

		
		if line_idx in valid_indices:
			valid_inputs.write('\n')
			valid_labels.write('\n')
		else:
			train_inputs.write('\n')
			train_labels.write('\n')
			

		


train_inputs.close()
train_labels.close()
valid_inputs.close()
valid_labels.close()



