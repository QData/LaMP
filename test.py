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




def test_epoch(model, test_data,opt,data_dict, description):
	model.eval()
	out_len = (opt.tgt_vocab_size)
	all_predictions = torch.zeros(len(test_data._src_insts),out_len)
	all_targets = torch.zeros(len(test_data._src_insts),out_len)
	batch_idx = 0
	batch_size = test_data._batch_size
	bce_total = 0

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

