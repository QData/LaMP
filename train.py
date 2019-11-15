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



def train_epoch(model,train_data, crit, optimizer,adv_optimizer,epoch,data_dict,opt):
	model.train()

	out_len = (opt.tgt_vocab_size) if opt.binary_relevance else (opt.tgt_vocab_size-1)

	all_predictions = torch.zeros(len(train_data._src_insts),out_len)
	all_targets = torch.zeros(len(train_data._src_insts),out_len)

	
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