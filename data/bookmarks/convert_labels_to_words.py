from pdb import set_trace as stop
import torch
import csv




# label_dict = torch.load('label_dict.pt')




# with open('test_labels_old.txt', newline='') as csvfile:
#     data = list(csv.reader(csvfile))

# out_file = open('test_labels.txt','w')

# for line in data:
# 	labels = line[0].split(' ')
# 	labels = list(map(int, labels))
# 	for label in labels[0:-1]:
# 		label_string = label_dict[label]
# 		out_file.write(label_string+' ')

# 	label_string = label_dict[labels[-1]]
# 	out_file.write(label_string+'\n')



input_dict = torch.load('inputs_dict.pt')

with open('test_inputs_old.txt', newline='') as csvfile:
    data = list(csv.reader(csvfile))

out_file = open('test_inputs.txt','w')

for line in data:
	labels = line[0].split(' ')
	labels = list(map(int, labels))
	for label in labels[0:-1]:
		label_string = input_dict[label]
		out_file.write(label_string+' ')

	label_string = input_dict[labels[-1]]
	out_file.write(label_string+'\n')
