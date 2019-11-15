import argparse,math,time,warnings,copy, numpy as np, os.path as path 
import evals,utils
import torch, torch.nn as nn, torch.nn.functional as F
import lamp.Constants as Constants
from lamp.Models import LAMP
from utils import LabelSmoothing
from lamp.Translator import translate
from lamp.DataLoader import process_data
from config_args import config_args,get_args
from pdb import set_trace as stop
from tqdm import tqdm
from evals import Logger
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
args = get_args(parser)
opt = config_args(args)



def train_epoch(model,train_data, crit, optimizer,adv_optimizer,epoch,data_dict,opt):
	model.train()

	out_len = (opt.tgt_vocab_size) if opt.binary_relevance else (opt.tgt_vocab_size-1)

	all_predictions = torch.zeros(len(train_data._src_insts),out_len)
	all_targets = torch.zeros(len(train_data._src_insts),out_len)

	if 'gene' in opt.dataset:
		idx_tensor = []
		for gene_name,gene_index in data_dict['tgt'].items():
			if gene_name in utils.labels105u:
				idx_tensor.append(gene_index-4)
		idx_tensor = torch.Tensor(idx_tensor).long()

	
	batch_idx,batch_size = 0,train_data._batch_size
	bce_total,d_total,d_fake_total,g_total = 0,0,0,0

	
	for batch in tqdm(train_data, mininterval=0.5,desc='(Training)', leave=False):
		src,adj,tgt = batch
		loss,d_loss = 0,0
		
		gold = tgt[:, 1:]

		if opt.binary_relevance:
			gold_binary = utils.get_gold_binary(gold.data.cpu(),opt.tgt_vocab_size).cuda()

			optimizer.zero_grad()

			pred,enc_output,*results = model(src,adj,None,gold_binary,return_attns=opt.attns_loss,int_preds=opt.int_preds)
			norm_pred = F.sigmoid(pred)


			bce_loss =  F.binary_cross_entropy_with_logits(pred, gold_binary,reduction='mean')
			loss += bce_loss
			bce_total += bce_loss.item()
			if opt.int_preds and not opt.matching_mlp:
				for i in range(len(results[0])):
					bce_loss =  F.binary_cross_entropy_with_logits(results[0][i], gold_binary,reduction='mean')
					loss += (opt.int_pred_weight)*bce_loss
				
			
			if epoch == opt.thresh1:
				opt.init_model = copy.deepcopy(model)

			loss.backward()
			optimizer.step()

			tgt_out = gold_binary.data
			pred_out = norm_pred.data


		else: 
			# Non Binary Outputs
			optimizer.zero_grad()
			pred,enc_output,*results = model(src,adj,tgt,None,int_preds=opt.int_preds)
			loss = crit(F.log_softmax(pred), gold.contiguous().view(-1))
			pred = F.softmax(pred,dim=1)
			pred_vals,pred_idxs = pred.max(1)
			pred_vals = pred_vals.view(gold.size()).data.cpu()
			pred_idxs = pred_idxs.view(gold.size()).data.cpu()
			pred_out = torch.zeros(pred_vals.size(0),pred.size(1)).scatter_(1,pred_idxs.long(),pred_vals)
			tgt_out = torch.zeros(pred_vals.size(0),pred.size(1)).scatter_(1,gold.data.cpu().long(),torch.ones(pred_vals.size()))
			pred_out = pred_out[:,1:]
			tgt_out = tgt_out[:,1:]
			loss.backward()
			optimizer.step()

		
		## Updates ##
		start_idx, end_idx = (batch_idx*batch_size),((batch_idx+1)*batch_size)
		all_predictions[start_idx:end_idx] = pred_out
		all_targets[start_idx:end_idx] = tgt_out
		batch_idx +=1
		
	
	return all_predictions, all_targets, bce_total


def test_epoch(model, test_data,opt,data_dict, description):
	model.eval()
	out_len = (opt.tgt_vocab_size)
	all_predictions = torch.zeros(len(test_data._src_insts),out_len)
	all_targets = torch.zeros(len(test_data._src_insts),out_len)

	batch_idx = 0
	batch_size = test_data._batch_size

	bce_total = 0

	if 'gene' in opt.dataset:
		idx_tensor = []
		for gene_name,gene_index in data_dict['tgt'].items():
			if gene_name in utils.labels105u:
				idx_tensor.append(gene_index-4)
		idx_tensor = torch.Tensor(idx_tensor).long()

	for batch in tqdm(test_data, mininterval=0.5, desc=description, leave=False):
		src,adj,tgt = batch
		batch_loc = int(batch_idx*batch_size)

		gold = tgt[:, 1:]
		
		if opt.binary_relevance:
			pad_batch = False
			if opt.multi_gpu and (batch[0][0].size(0) < opt.batch_size): 
				pad_batch = True

			if pad_batch:
				diff = opt.batch_size - src[0].size(0)
				src = [torch.cat((src[0],torch.zeros(diff,src[0].size(1)).type(src[0].type()).cuda()),0),
					   torch.cat((src[1],torch.zeros(diff,src[1].size(1)).type(src[1].type()).cuda()),0)]
				tgt = torch.cat((tgt,torch.zeros(diff,tgt.size(1)).type(tgt.type()).cuda()),0)
				
			pred,enc_output,*results = model(src,adj, None, None,int_preds=opt.int_preds)

			

			if pad_batch:
				pred = pred[0:batch[0][0].size(0)]
				gold = gold[0:batch[0][0].size(0)]
				
			gold_binary = utils.get_gold_binary(gold.data.cpu(),opt.tgt_vocab_size).cuda()

			norm_pred = F.sigmoid(pred).data

			bce_loss =  F.binary_cross_entropy_with_logits(pred, gold_binary,reduction='mean')
			bce_total += bce_loss.item()

						
			start_idx, end_idx = (batch_idx*batch_size),((batch_idx+1)*batch_size)
			all_predictions[start_idx:end_idx] = norm_pred
			all_targets[start_idx:end_idx] = gold_binary

		else:
			all_hyp, all_scores = translate(model,opt,src,adj)		
			for sample_idx,pred in enumerate(all_hyp):
				pred = pred[0]
				for label_idx,label in enumerate(pred):
					if label == Constants.EOS:
						break
					elif label != Constants.PAD and label != Constants.BOS: 
						all_predictions[batch_loc+sample_idx][label] = all_scores[sample_idx][label_idx]
			gold = tgt[:, 1:].data
			for sample_idx,labels in enumerate(gold):
				for label in labels:
					if label == Constants.EOS:
						break
					elif label != Constants.PAD and label != Constants.BOS:
						all_targets[batch_loc+sample_idx][label] = 1
			
		batch_idx+=1
	

	
	return all_predictions, all_targets, bce_total


def run_model(model, train_data, valid_data, test_data, crit, optimizer,adv_optimizer,scheduler, opt, data_dict):
	logger = Logger(opt)
	
	valid_accus = []

	losses = []

	if opt.test_only:
		start = time.time()
		all_predictions, all_targets, test_loss = test_epoch(model, test_data,opt,data_dict,'(Testing)')
		elapsed = ((time.time()-start)/60)
		print('\n(Testing) elapse: {elapse:3.3f} min'.format(elapse=elapsed))
		test_loss = test_loss/len(test_data._src_insts)
		print('B : '+str(test_loss))

		test_metrics = evals.compute_metrics(all_predictions,all_targets,0,opt,elapsed,all_metrics=True)

		return

	loss_file = open(path.join(opt.model_name,'losses.csv'),'w+')
	for epoch_i in range(opt.epoch):
		print('================= Epoch', epoch_i+1, '=================')
		if scheduler and opt.lr_decay > 0: scheduler.step()


		################################## TRAIN ###################################
		start = time.time()
		all_predictions,all_targets,train_loss=train_epoch(model,train_data,crit,optimizer,adv_optimizer,(epoch_i+1),data_dict,opt)
		elapsed = ((time.time()-start)/60)
		print('\n(Training) elapse: {elapse:3.3f} min'.format(elapse=elapsed))
		train_loss = train_loss/len(train_data._src_insts)
		print('B : '+str(train_loss))


		if 'reuters' in opt.dataset or 'bibtext' in opt.dataset:
			torch.save(all_predictions,path.join(opt.model_name,'epochs','train_preds'+str(epoch_i+1)+'.pt'))
			torch.save(all_targets,path.join(opt.model_name,'epochs','train_targets'+str(epoch_i+1)+'.pt'))
		train_metrics = evals.compute_metrics(all_predictions,all_targets,0,opt,elapsed,all_metrics=True)  

		################################### VALID ###################################
		start = time.time()
		all_predictions, all_targets,valid_loss = test_epoch(model, valid_data,opt,data_dict,'(Validation)')
		elapsed = ((time.time()-start)/60)
		print('\n(Validation) elapse: {elapse:3.3f} min'.format(elapse=elapsed))
		valid_loss = valid_loss/len(valid_data._src_insts)
		print('B : '+str(valid_loss))

		torch.save(all_predictions,path.join(opt.model_name,'epochs','valid_preds'+str(epoch_i+1)+'.pt'))
		torch.save(all_targets,path.join(opt.model_name,'epochs','valid_targets'+str(epoch_i+1)+'.pt'))
		valid_metrics = evals.compute_metrics(all_predictions,all_targets,0,opt,elapsed,all_metrics=True)
		valid_accu = valid_metrics['ACC']
		valid_accus += [valid_accu]

		################################## TEST ###################################
		start = time.time()
		all_predictions, all_targets, test_loss = test_epoch(model, test_data,opt,data_dict,'(Testing)')
		elapsed = ((time.time()-start)/60)
		print('\n(Testing) elapse: {elapse:3.3f} min'.format(elapse=elapsed))
		test_loss = test_loss/len(test_data._src_insts)
		print('B : '+str(test_loss))

		torch.save(all_predictions,path.join(opt.model_name,'epochs','test_preds'+str(epoch_i+1)+'.pt'))
		torch.save(all_targets,path.join(opt.model_name,'epochs','test_targets'+str(epoch_i+1)+'.pt'))
		test_metrics = evals.compute_metrics(all_predictions,all_targets,0,opt,elapsed,all_metrics=True)
		
		best_valid,best_test = logger.evaluate(train_metrics,valid_metrics,test_metrics,epoch_i,opt.total_num_parameters)

		print(opt.model_name)

		losses.append([epoch_i+1,train_loss,valid_loss,test_loss])
		
		if not 'test' in opt.model_name and not opt.test_only:
			utils.save_model(opt,epoch_i,model,valid_accu,valid_accus)



		loss_file.write(str(int(epoch_i+1)))
		loss_file.write(','+str(train_loss))
		loss_file.write(','+str(valid_loss))
		loss_file.write(','+str(test_loss))
		loss_file.write('\n')




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
