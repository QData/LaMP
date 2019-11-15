import argparse,math,time,warnings,copy, numpy as np, os.path as path 
import utils.evals as evals
import utils.utils as utils
from utils.data_loader import process_data
import torch, torch.nn as nn, torch.nn.functional as F
import lamp.Constants as Constants
from lamp.Models import LAMP
from lamp.Translator import translate
from config_args import config_args,get_args
from pdb import set_trace as stop
from tqdm import tqdm
from runner import run_model
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
args = get_args(parser)
opt = config_args(args)



def main(opt):
	#========= Loading Dataset =========#
	data = torch.load(opt.data)
	vocab_size = len(data['dict']['tgt'])
	
	global_labels = None
	for i in range(len(data['train']['src'])):
		labels = torch.tensor(data['train']['tgt'][i]).unsqueeze(0)
		labels = utils.get_gold_binary_full(labels,vocab_size)
		if global_labels is None:
			global_labels = labels
		else:
			global_labels+=labels

	for i in range(len(data['valid']['src'])):
		labels = torch.tensor(data['valid']['tgt'][i]).unsqueeze(0)
		labels = utils.get_gold_binary_full(labels,vocab_size)
		global_labels+=labels
		
	for i in range(len(data['test']['src'])):
		labels = torch.tensor(data['test']['tgt'][i]).unsqueeze(0)
		labels = utils.get_gold_binary_full(labels,vocab_size)
		global_labels+=labels

	global_labels = global_labels[0][0:-4]

	ranked_labels,ranked_idx = torch.sort(global_labels)

	indices = ranked_idx[2:24].long()
	label_count = ranked_labels[2:24]


	train_data,valid_data,test_data,label_adj_matrix,opt = process_data(data,opt)
	print(opt)

	#========= Preparing Model =========#
	model = LAMP(
		opt.src_vocab_size,
		opt.tgt_vocab_size,
		opt.max_token_seq_len_e,
		opt.max_token_seq_len_d,
		proj_share_weight=opt.proj_share_weight,
		embs_share_weight=opt.embs_share_weight,
		d_k=opt.d_k,
		d_v=opt.d_v,
		d_model=opt.d_model,
		d_word_vec=opt.d_word_vec,
		d_inner_hid=opt.d_inner_hid,
		n_layers_enc=opt.n_layers_enc,
		n_layers_dec=opt.n_layers_dec,
		n_head=opt.n_head,
		n_head2=opt.n_head2,
		dropout=opt.dropout,
		dec_dropout=opt.dec_dropout,
		dec_dropout2=opt.dec_dropout2,
		encoder=opt.encoder,
		decoder=opt.decoder,
		enc_transform=opt.enc_transform,
		onehot=opt.onehot,
		no_enc_pos_embedding=opt.no_enc_pos_embedding,
		no_dec_self_att=opt.no_dec_self_att,
		loss=opt.loss,
		label_adj_matrix=label_adj_matrix,
		attn_type=opt.attn_type,
		label_mask=opt.label_mask,
		matching_mlp=opt.matching_mlp,
		graph_conv=opt.graph_conv,
		int_preds=opt.int_preds)

	print(model)
	print(opt.model_name)


	opt.total_num_parameters = int(utils.count_parameters(model))

	if opt.load_emb:
		model = utils.load_embeddings(model,'../../Data/word_embedding_dict.pth')
 
	optimizer = torch.optim.Adam(model.get_trainable_parameters(),betas=(0.9, 0.98),lr=opt.lr)
	scheduler = torch.torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_decay,last_epoch=-1)

	adv_optimizer = None
	
	crit = utils.get_criterion(opt)

	if torch.cuda.device_count() > 1 and opt.multi_gpu:
		print("Using", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)

	if torch.cuda.is_available() and opt.cuda:
		model = model.cuda()
	
		crit = crit.cuda()
		if opt.gpu_id != -1:
			torch.cuda.set_device(opt.gpu_id)

	if opt.load_pretrained:		
		checkpoint = torch.load(opt.model_name+'/model.chkpt')
		model.load_state_dict(checkpoint['model'])

	try:
		run_model(model,train_data,valid_data,test_data,crit,optimizer, adv_optimizer,scheduler,opt,data['dict'])
	except KeyboardInterrupt:
		print('-' * 89+'\nManual Exit')
		exit()

if __name__ == '__main__':
	main(opt)
