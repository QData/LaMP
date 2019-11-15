from nltk.corpus import wordnet as wn
import os
import torch
import copy

dataset = 'bibtext'
path = os.path.join(dataset,'labels.txt')


new_dict = {'<s>': 2, '</s>': 3, '<blank>': 0, '<unk>': 1}


output_path = os.path.join(dataset,'tf_interactions.tsv')
out_file = open(output_path,'w+') 

out_file.write('#node1\tnode2\n')

labels = [line.rstrip('\n') for line in open(path)]


label_dict = {}
k = 1
for label in labels:
	label_dict[str(k)] = label
	k+=1


# label_dict = {'1': '20',
 # '10': 'article',
 # '100': 'links',
 # '101': 'linux',

# old_dict = {{'1': 152,
 # '10': 91,
 # '100': 139,
 # '101': 38,
 #  '99': 109,
 # '</s>': 3,
 # '<blank>': 0,
 # '<s>': 2,
 # '<unk>': 1}



data = torch.load(os.path.join(dataset,'train_valid_test_old.pt'))
old_dict = data['dict']['tgt']
for k,v in label_dict.items():
	if k in old_dict.keys():
		old_dict['*****'+v] = old_dict[k]
		del old_dict[k]


kv_list = copy.copy(old_dict)
for k,v in kv_list.items():
	if '*****' in k:
		key_repl = k.replace('*****','')
		old_dict[key_repl] = old_dict[k]
		del old_dict[k]


data['dict']['tgt'] = old_dict
torch.save(data,os.path.join(dataset,'train_valid_test.pt'))




	

