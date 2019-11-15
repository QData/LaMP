import torch
import numpy as np
import scipy.io as sio
from pdb import set_trace as stop
data = torch.load('train_valid_test.pt')


mat_dict = {
'X1':[],
'X2':[],
'XTe1':[],
'XTe2':[],
'XV1':[],
'XV2':[]
}


for sample_input, sample_label in zip(data['train']['src'], data['train']['tgt']):
	sample_input = (torch.tensor(sample_input[1:-1])-4).long()
	sample_label = (torch.tensor(sample_label[1:-1])-4).long()

	sample_input_binary = torch.zeros(len(data['dict']['src'])-4).index_fill_(0, sample_input, 1).numpy().tolist()
	sample_label_binary = torch.zeros(len(data['dict']['tgt'])-4).index_fill_(0, sample_label, 1).numpy().tolist()

	mat_dict['X1'].append(sample_input_binary)
	mat_dict['X2'].append(sample_label_binary)

mat_dict['X1'] = np.float32(np.asarray(mat_dict['X1']))
mat_dict['X2'] = np.float32(np.asarray(mat_dict['X2']))



for sample_input, sample_label in zip(data['valid']['src'], data['valid']['tgt']):
	sample_input = (torch.tensor(sample_input[1:-1])-4).long()
	sample_label = (torch.tensor(sample_label[1:-1])-4).long()

	sample_input_binary = torch.zeros(len(data['dict']['src'])-4).index_fill_(0, sample_input, 1).numpy().tolist()
	sample_label_binary = torch.zeros(len(data['dict']['tgt'])-4).index_fill_(0, sample_label, 1).numpy().tolist()

	mat_dict['XV1'].append(sample_input_binary)
	mat_dict['XV2'].append(sample_label_binary)

mat_dict['XV1'] = np.float32(np.asarray(mat_dict['XV1']))
mat_dict['XV2'] = np.float32(np.asarray(mat_dict['XV2']))



for sample_input_i, sample_label_i in zip(data['test']['src'], data['test']['tgt']):
	sample_input = (torch.tensor(sample_input_i[1:-1])-4).long()
	sample_label = (torch.tensor(sample_label_i[1:-1])-4).long()

	sample_input[sample_input<0] = 0 # handles unknown

	sample_input_binary = torch.zeros(len(data['dict']['src'])-4).index_fill_(0, sample_input, 1).numpy().tolist()
	try:
		sample_label_binary = torch.zeros(len(data['dict']['tgt'])-4).index_fill_(0, sample_label, 1).numpy().tolist()
		mat_dict['XTe1'].append(sample_input_binary)
		mat_dict['XTe2'].append(sample_label_binary)
	except:
		pass

mat_dict['XTe1'] = np.float32(np.asarray(mat_dict['XTe1']))
mat_dict['XTe2'] = np.float32(np.asarray(mat_dict['XTe2']))



sio.savemat('delicious.mat', mat_dict)