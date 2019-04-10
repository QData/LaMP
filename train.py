import argparse,math,time,warnings,copy, numpy as np, os.path as path 
import evals,utils
import torch, torch.nn as nn, torch.nn.functional as F
import models.Constants as Constants
from models.Models import Transformer
from models.Discriminators import Discriminator
from models.Optim import ScheduledOptim
from utils import LabelSmoothing
from models.Translator import translate
from data_loader import process_data
from config_args import config_args,get_args
from pdb import set_trace as stop
from tqdm import tqdm
from evals import Logger
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
args = get_args(parser)
opt = config_args(args)

# forward_tree = torch.load('/af11/jjl5sw/deepENCODE/data/gene/forward_tree.pt')


# CUDA_VISIBLE_DEVICES=1 python train.py -dataset sider -save_mode best -batch_size 32 -d_model 512 -d_inner_hid 512 -n_layers_enc 2 -n_layers_dec 2 -n_head 4 -epoch 50 -dropout 0.2 -dec_dropout 0.2 -lr 0.0002 -encoder 'graph' -decoder 'graph' -load_pretrained -test_only

#CUDA_VISIBLE_DEVICES=1 python train.py -dataset bibtext -save_mode best -batch_size 32 -d_model 512 -d_inner_hid 512 -n_layers_enc 3 -n_layers_dec 3 -n_head 4 -epoch 50 -dropout 0.2 -dec_dropout 0.2 -lr 0.0002 -encoder 'graph' -decoder 'graph' -enc_transform ''



def train_epoch(model, discriminator,train_data, crit, optimizer,adv_optimizer,epoch,data_dict,opt):
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

			############ Generator ##############
			optimizer.zero_grad()


			pred,enc_output,*results = model(src,adj,None,gold_binary,return_attns=opt.attns_loss,int_preds=opt.int_preds)
			norm_pred = F.sigmoid(pred)

			if opt.attns_loss:
				dec_self_attns = results[1][0]
				
				attns0 = dec_self_attns[0].view(tgt.size(0),-1,dec_self_attns[0].size(-2),dec_self_attns[0].size(-1))
				attns1 = dec_self_attns[1].view(tgt.size(0),-1,dec_self_attns[1].size(-2),dec_self_attns[1].size(-1))
				# attns2 = dec_self_attns[2].view(tgt.size(0),-1,dec_self_attns[2].size(-2),dec_self_attns[2].size(-1))

				mask = torch.ones_like(attns0[0][0]) - torch.diag(torch.ones_like(attns0[0][0][0])).repeat(tgt.size(0),1,1)

				attns_true = gold_binary.repeat(gold_binary.size(1),1).view(tgt.size(0),opt.tgt_vocab_size,opt.tgt_vocab_size)*mask
				attns_true = attns_true.unsqueeze(1).repeat(1,attns0.size(1),1,1)*gold_binary.sum(1).view(-1,1,1,1)

				attn_loss0 = torch.nn.functional.mse_loss(attns0,attns_true,reduction='mean')
				attn_loss1 = torch.nn.functional.mse_loss(attns1,attns_true,reduction='mean')
				# attn_loss2 = torch.nn.functional.mse_loss(attns2,attns_true,reduction='mean')

				loss += 0.2*(attn_loss0)
				loss += 0.2*(attn_loss1)

			if opt.loss == 'adv' and epoch > opt.thresh1: 
				if opt.adv_type == 'infnet':
					delta = F.pairwise_distance(norm_pred,gold_binary)
					g_fake_out = discriminator(src,adj,norm_pred)
					g_real_out = discriminator(src,adj,gold_binary)
					g_loss = torch.mean(F.relu(-delta+g_fake_out-g_real_out))
				else:
					g_fake_out = discriminator(src,adj,norm_pred)
					g_loss = F.binary_cross_entropy_with_logits(g_fake_out, torch.ones(g_fake_out.size()).cuda(),reduction='mean')

				loss += g_loss
				g_total += g_loss.item()

				reg_losses = [F.mse_loss(params2,params1,reduction='sum') for params1, params2 in zip(opt.init_model.parameters(),model.parameters())]
				reg_loss = sum(reg_losses)
				loss += reg_loss

				if opt.bce_with_adv:
					bce_loss =  F.binary_cross_entropy_with_logits(pred, gold_binary,reduction='mean')
					bce_total += bce_loss.item()
					loss += bce_loss
								
			else:
				if opt.loss == 'ranking':
					# loss += F.binary_cross_entropy_with_logits(pred, gold_binary,reduction='mean')
					# margin_loss = torch.nn.MultiLabelMarginLoss(size_average=None, reduce=None, reduction='mean')

					# gold_binary_sorted,sorted_indices = torch.sort(gold_binary, dim=1, descending=True, out=None)
					# norm_pred_sorted = torch.index_select(norm_pred.view(-1), 0, sorted_indices.view(-1)).view(norm_pred.shape)

					# loss += margin_loss(norm_pred_sorted,gold_binary_sorted.long())

					pos_size = gold_binary.sum(1)
					neg_size = gold_binary.size(1) - gold_binary.sum(1)
					normalizer = 1/(pos_size*neg_size)

					for idx in range(norm_pred.size(0)):
						pos_elements = gold_binary[idx].nonzero().detach()
						neg_elements = (gold_binary[idx] == 0).nonzero().detach().view(-1)
						neg_pred = torch.index_select(norm_pred[idx], 0, neg_elements)

						ranking_loss = 0
						for pos_idx in pos_elements:
							pos_pred = norm_pred[idx][pos_idx.item()].view(-1).repeat(len(neg_pred))
							exp_loss = torch.exp(-(neg_pred-pos_pred))
							ranking_loss += exp_loss.sum()

						loss -= normalizer[idx]*ranking_loss

				else:
					bce_loss =  F.binary_cross_entropy_with_logits(pred, gold_binary,reduction='mean')
					loss += bce_loss
					bce_total += bce_loss.item()
					if opt.int_preds and not opt.matching_mlp:
						for i in range(len(results[0])):
							bce_loss =  F.binary_cross_entropy_with_logits(results[0][i], gold_binary,reduction='mean')
							loss += (opt.int_pred_weight)*bce_loss
					

				if opt.matching_mlp:
					matching_loss = F.mse_loss(results[0][0][0],results[0][1][0],reduction='mean')
					loss += 0.2*matching_loss
					matching_loss = F.mse_loss(results[0][0][1],results[0][1][1],reduction='mean')
					loss += 0.2*matching_loss
					# matching_loss = F.mse_loss(results[0][0],results[0][1],reduction='mean')
					# loss += matching_loss

				if opt.loss2 == 'l2':
					l2_loss = torch.mean(F.pairwise_distance(pred, gold_binary))
					loss += l2_loss
				elif opt.loss2 == 'kl':
					kl_loss = torch.mean(F.kl_div(torch.log(F.sigmoid(pred)), gold_binary))
					loss += kl_loss
				
				if epoch == opt.thresh1:
					opt.init_model = copy.deepcopy(model)

			loss.backward()
			optimizer.step()

			tgt_out = gold_binary.data
			pred_out = norm_pred.data

			########## Discriminator ############
			if opt.loss == 'adv' and epoch > opt.thresh1:
				adv_optimizer.zero_grad()

				norm_pred = norm_pred.detach()
				pred = pred.detach()

				if opt.adv_type == 'infnet':
					delta = F.pairwise_distance(norm_pred,gold_binary)
					d_fake_out = discriminator(src,adj,norm_pred)
					d_real_out = discriminator(src,adj,gold_binary)				
					d_loss = torch.mean(F.relu(delta-d_fake_out+d_real_out))
					d_total += d_loss.item()
					
					# alpha = torch.rand(gold_binary.size()).cuda()
					# x_hat = (alpha * gold_binary.data) + ((1 - alpha) * norm_pred.requires_grad_(True))
					# alpha_out = discriminator(src,adj,x_hat)
					# d_loss_gp = utils.gradient_penalty(alpha_out, x_hat)
					# d_loss += 1*d_loss_gp

				else:
					d_real_out = discriminator(src,adj,gold_binary)
					d_fake_out = discriminator(src,adj,norm_pred)
					d_real_loss= 0.5*F.binary_cross_entropy_with_logits(d_real_out, torch.ones(d_real_out.size()).cuda())
					d_fake_loss =  0.5*F.binary_cross_entropy_with_logits(d_fake_out, torch.zeros(d_fake_out.size()).cuda())
					d_loss += d_real_loss + d_fake_loss
					d_total += (d_real_loss.item()+d_fake_loss.item())
				
				d_loss.backward()
				adv_optimizer.step()



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
		

	# print('B : '+str(bce_total))
	if opt.loss == 'adv': print('D : '+str(d_total)+'\nG : '+str(g_total))
	
	

	return all_predictions, all_targets, bce_total


def test_epoch(model, test_data,opt,data_dict, description):
	# stop()
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
	
	# if 'gene' in opt.dataset:
	#	 for parent in forward_tree:
	#		 parent_idx = data_dict['tgt'][parent]-4
	#		 for child in forward_tree[parent]:
	#			 child_idx = data_dict['tgt'][child]-4
	#			 all_targets[:,child_idx][all_targets[:,child_idx]<1] = float('NaN')
	#			 all_predictions[:,child_idx][all_predictions[:,child_idx]<1] = float('NaN')
		
	#	 all_predictions = torch.index_select(all_predictions, 1, idx_tensor)
	#	 all_targets = torch.index_select(all_targets, 1, idx_tensor)
	#	 print('**')
	
	return all_predictions, all_targets, bce_total


def run_model(model,discriminator, train_data, valid_data, test_data, crit, optimizer,adv_optimizer,scheduler, opt, data_dict):
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

		# for param in model.decoder.layer_stack[0].parameters(): param.requires_grad = False

		################################## TRAIN ###################################
		start = time.time()
		all_predictions,all_targets,train_loss=train_epoch(model,discriminator,train_data,crit,optimizer,adv_optimizer,(epoch_i+1),data_dict,opt)
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
	# losses = np.asarray(losses).astype(float)
	# print(path.join(opt.model_name,'losses.csv'))
	# np.savetxt(path.join(opt.model_name,'losses.csv'), losses, delimiter=",")


def get_small_tfs(data,indices,rev_dict_src,rev_dict_tgt):
	src_file = open('small_tf_inputs.txt','w')
	tgt_file = open('small_tf_labels.txt','w')

	for i in range(len(data['train']['src'])):
		src = data['train']['src'][i][1:-1]
		labels = data['train']['tgt'][i][1:-1]

		Flag = False
		for label in labels:
			if label in indices:
				Flag = True
				tgt_file.write(rev_dict_tgt[label]+' ')
		if Flag is True:
			tgt_file.write('\n')
			for feature in src:
				src_file.write(rev_dict_src[feature]+' ')
			src_file.write('\n')

	for i in range(len(data['valid']['src'])):
		src = data['valid']['src'][i][1:-1]
		labels = data['valid']['tgt'][i][1:-1]
		Flag = False
		for label in labels:
			if label in indices:
				Flag = True
				tgt_file.write(rev_dict_tgt[label]+' ')
		if Flag is True:
			tgt_file.write('\n')
			for feature in src:
				src_file.write(rev_dict_src[feature]+' ')
			src_file.write('\n')
		
	for i in range(len(data['test']['src'])):
		src = data['test']['src'][i][1:-1]
		labels = data['test']['tgt'][i][1:-1]
		Flag = False
		for label in labels:
			if label in indices:
				Flag = True
				tgt_file.write(rev_dict_tgt[label]+' ')
		if Flag is True:
			tgt_file.write('\n')
			for feature in src:
				src_file.write(rev_dict_src[feature]+' ')
			src_file.write('\n')

	return None


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

	# for i in range(len(indices)): print(rev_dict_tgt[indices[i].item()].upper()+','+str(int(label_count[i].item())))
	
	# rev_dict_src = {v: k for k, v in data['dict']['src'].items()}
	# rev_dict_tgt = {v: k for k, v in data['dict']['tgt'].items()}

	# get_small_tfs(data,indices,rev_dict_src,rev_dict_tgt)

	# ranked_labels2 = torch.index_select(global_labels, 0, indices)

	train_data,valid_data,test_data,label_adj_matrix,opt = process_data(data,opt)
	print(opt)

	#========= Preparing Model =========#
	
	
	discriminator = None
	if opt.loss == 'adv':
		discriminator = Discriminator(
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
			attn_type=opt.attn_type,
			no_enc_pos_embedding=opt.no_enc_pos_embedding,
			no_dec_self_att=opt.no_dec_self_att,
			loss=opt.loss,
			label_adj_matrix=label_adj_matrix,
			label_mask=opt.label_mask,
			int_preds=opt.int_preds)
		print(discriminator)

	transformer = Transformer(
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

	print(transformer)
	print(opt.model_name)


	opt.total_num_parameters = int(utils.count_parameters(transformer))

	# pretrained_embeddings = pickle.load(open("Data/word_embedding_dict.pt","rb"), encoding='iso-8859-1' )

	if opt.load_emb:
		transformer = utils.load_embeddings(transformer,'../../Data/word_embedding_dict.pth')
 
	optimizer = torch.optim.Adam(transformer.get_trainable_parameters(),betas=(0.9, 0.98),lr=opt.lr)
	scheduler = torch.torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_decay,last_epoch=-1)

	adv_optimizer = None
	if opt.loss == 'adv':
		adv_optimizer = torch.optim.Adam(discriminator.parameters(),betas=(0.9, 0.98),lr=opt.lr)
	
	crit = utils.get_criterion(opt)

	if torch.cuda.device_count() > 1 and opt.multi_gpu:
		print("Using", torch.cuda.device_count(), "GPUs!")
		transformer = nn.DataParallel(transformer)
		# if opt.matching_mlp: 
		#	 transformer.matching_network = nn.DataParallel(transformer.matching_network)
		if discriminator:
			discriminator = nn.DataParallel(discriminator)
	if torch.cuda.is_available() and opt.cuda:
		transformer = transformer.cuda()
		
		if discriminator:
			discriminator = discriminator.cuda()
		crit = crit.cuda()
		if opt.gpu_id != -1:
			torch.cuda.set_device(opt.gpu_id)

	if opt.load_pretrained:		
		checkpoint = torch.load(opt.model_name+'/model.chkpt')
		transformer.load_state_dict(checkpoint['model'])

	try:
		run_model(transformer,discriminator,train_data,valid_data,test_data,crit,optimizer, adv_optimizer,scheduler,opt,data['dict'])
	except KeyboardInterrupt:
		print('-' * 89+'\nManual Exit')
		exit()

if __name__ == '__main__':
	main(opt)
