import argparse
import math
import time
# from pdb import set_trace as stop
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import models.Constants as Constants
from models.Models import Transformer
from models.Optim import ScheduledOptim
from data_loader import DataLoader
import numpy
import numpy as np
import os
import evals
from models.Beam import Beam
from models.Translator import translate
import warnings
import os.path as path 
from evals import Logger
from utils import get_pairwise_adj
from IPython.core.debugger import set_trace as stop
import utils
from config_args import config_args,get_args
import operator
from data_loader import process_data
import sklearn.preprocessing as preprocessing
import sklearn.cluster as cluster



import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore")



# CUDA_VISIBLE_DEVICES=0 python visualize.py -dataset bookmarks -save_mode best -batch_size 32 -d_model 512 -d_inner_hid 512 -n_layers_enc 2 -n_layers_dec 2 -n_head 4 -epoch 50 -dropout 0.1 -dec_dropout 0.1 -lr 0.0002 -encoder 'graph' -decoder 'graph' -label_mask 'prior' -int_preds -viz



get_enc_self_attns = False
get_enc_dec_attns = True
get_dec_self_attns = True


parser = argparse.ArgumentParser()
parser.add_argument('-one_sample', action='store_true')
parser.add_argument('-save_root', type=str, default='visualizations')
parser.add_argument('-normalize', action='store_true')
parser.add_argument('-shuffle', action='store_true')
args = get_args(parser)
opt = config_args(args)



if opt.n_layers_dec is None:
	opt.n_layers_dec = opt.n_layers_enc

opt.epoch = 50
if opt.decoder in ['sa_m','pmlp']:
	opt.proj_share_weight = True
else:
	opt.proj_share_weight = False

if opt.dataset in ['deepsea','gm12878','gm12878_unique2','gm12878_unique']:
	opt.onehot=True


if opt.test_batch_size <= 0:
	opt.test_batch_size = opt.batch_size

if opt.d_v == -1:
	opt.d_v = int(opt.d_model/opt.n_head)
if opt.d_k == -1:
	opt.d_k = int(opt.d_model/opt.n_head)

if opt.dec_dropout == -1:
	opt.dec_dropout = opt.dropout


if opt.dataset in ['bibtext','delicious','bookmarks']:
	opt.no_enc_pos_embedding = True
elif opt.dataset == 'bookmarks':
	opt.max_encoder_len = 500
	opt.max_ar_length = 48

if opt.d_inner_hid == -1:
	opt.d_inner_hid = int(opt.d_model*2)







opt.cuda = True
opt.save_dir = path.join(opt.save_root,opt.data_type)
if not os.path.exists(opt.save_dir):
	os.makedirs(opt.save_dir)


opt.results_dir = path.join('results',opt.data_type)
opt.data = path.join(opt.dataset,'train_valid_test.pt')



opt.mname='enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.priormask.int_preds_02'

# opt.mname='enc_sa.dec_pmlp.512.1024.128.128.nlayers_3_3.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.dropout_20_20.dec_reverse.no_residual'

if (opt.adj_matrix_lambda > 0):
	label_adj_matrix = get_pairwise_adj(data['dict']['tgt'],path.join(opt.dataset,'tf_interactions.tsv'))
else:
	label_adj_matrix = None


opt.no_residual = True

data = torch.load(opt.data)


train_data,valid_data,test_data,label_adj_matrix,opt = process_data(data,opt)

data_object = DataLoader(
	data['dict']['src'],
	data['dict']['tgt'],
	src_insts=data['train']['src'],
	tgt_insts=data['train']['tgt'],
	batch_size=opt.batch_size,
	shuffle=opt.shuffle,
	cuda=opt.cuda)



# data_object = DataLoader(
# 	data['dict']['src'],
# 	data['dict']['tgt'],
# 	src_insts=data['test']['src'],
# 	tgt_insts=data['test']['tgt'],
# 	batch_size=opt.batch_size,
# 	shuffle=opt.shuffle,
# 	cuda=opt.cuda)


opt.src_vocab_size = data_object.src_vocab_size
opt.tgt_vocab_size = data_object.tgt_vocab_size
opt.tgt_vocab_size = opt.tgt_vocab_size - 4

opt.d_v = int(opt.d_model/opt.n_head)
opt.d_k = int(opt.d_model/opt.n_head)
opt.max_token_seq_len_e = data['settings'].max_seq_len
opt.max_token_seq_len_d = 30
opt.proj_share_weight = True
opt.d_word_vec = opt.d_model



label_dict = {'ccat':'corporate/industrial','c11':'strategy/plans','c12':'legal/judicial','c13':'regulation/policy','c14':'share listings','c15':'performance','c151':'accounts/earnings','c1511':'annual results','c152':'comment/forecasts','c16':'insolvency/liquidity','c17':'funding/capital','c171':'share capital','c172':'bonds/debt issues','c173':'loans/credits','c174':'credit ratings','c18':'ownership changes','c181':'mergers/acquisitions','c182':'asset transfers','c183':'privatisations','c21':'production/services','c22':'new products/services','c23':'research/development','c24':'capacity/facilities','c31':'markets/marketing','c311':'domestic markets','c312':'external markets','c313':'market share','c32':'advertising/promotion','c33':'contracts/orders','c331':'defence contracts','c34':'monopolies/competition','c41':'management','c411':'management moves','c42':'labour','ecat':'economics','e11':'economic performance','e12':'monetary/economic','e121':'money supply','e13':'inflation/prices','e131':'consumer prices','e132':'wholesale prices','e14':'consumer finance','e141':'personal income','e142':'consumer credit','e143':'retail sales','e21':'government finance','e211':'expenditure/revenue','e212':'government borrowing','e31':'output/capacity','e311':'industrial production','e312':'capacity utilization','e313':'inventories','e41':'employment/labour','e411':'unemployment','e51':'trade/reserves','e511':'balance of payments','e512':'merchandise trade','e513':'reserves','e61':'housing starts','e71':'leading indicators','gcat':'government/social','g15':'european community','g151':'ec internal market','g152':'ec corporate policy','g153':'ec agriculture policy','g154':'ec monetary/economic','g155':'ec institutions','g156':'ec environment issues','g157':'ec competition/subsidy','g158':'ec external relations','g159':'ec general','gcrim':'crime/law enforcement','gdef':'defence','gdip':'international relations','gdis':'disasters and accidents','gent':'arts/culture/entertainment','genv':'environment and natural world','gfas':'fashion','ghea':'health','gjob':'labour issues','gmil':'millennium issues','gobit':'obituaries','godd':'human interest','gpol':'domestic politics','gpro':'biographies/personalities/people','grel':'religion','gsci':'science and technology','gspo':'sports','gtour':'travel and tourism','gvio':'war/civil war','gvote':'elections','gwea':'weather','gwelf':'welfare/social services','mcat':'markets','g15':'european community','gcrim':'crime/law enforcement','gdef':'defence','gdip':'international relations','gdis':'disasters and accidents','gent':'arts/culture/entertainment','genv':'environment and natural world','gfas':'fashion','ghea':'health','gjob':'labour issues','gmil':'millennium issues','gobit':'obituaries','godd':'human interest','gpol':'domestic politics','gpro':'biographies/personalities/people','grel':'religion','gsci':'science and technology','gspo':'sports','gtour':'travel and tourism','gvio':'war/civil war','gvote':'elections','gwea':'weather','gwelf':'welfare/social services','m11':'equity markets','m12':'bond markets','m13':'money markets','m131':'interbank markets','m132':'forex markets','m14':'commodity markets','m141':'soft commodities','m142':'metals trading','m143':'energy markets'}


	# label_dict = {'pou2f2':'pou2f2','rela':'rela','bcl3':'bcl3','maz':'maz','egr1':'egr1','foxm1':'foxm1','tcf7l1':'tcf7l1','stat1':'stat1','brca1':'brca1','rcor1':'rcor1','taf1':'taf1','nfe2':'nfe2','znf143':'znf143','tbl1xr1':'tbl1xr1','mxi1':'mxi1','bclaf1':'bclaf1','polr3g':'polr3g','myc':'myc','rxra':'rxra','stat5a':'stat5a','cebpb':'cebpb','znf274':'znf274','max':'max','e2f4':'e2f4','nr2c2':'nr2c2','stag1':'stag1','runx3':'runx3','ebf1':'ebf1','mef2c':'mef2c','chd2':'chd2','atf2':'atf2','usf1':'usf1','rest':'rest','sin3a':'sin3a','zzz3':'zzz3','mafk':'mafk','znf384':'znf384','brd4':'brd4','bcl11a':'bcl11a','usf2':'usf2','bhlhe40':'bhlhe40','gabpa':'gabpa','pml':'pml','mta3':'mta3','zeb1':'zeb1','nrf1':'nrf1','six5':'six5','stat3':'stat3','smc3':'smc3','jund':'jund','tp53':'tp53','irf3':'irf3','chd1':'chd1','ikzf1':'ikzf1','rad21':'rad21','atf3':'atf3','tbp':'tbp','mef2a':'mef2a','srebf1':'srebf1','yy1':'yy1','fos':'fos','esrra':'esrra','polr3a':'polr3a','pax5':'pax5','ets1':'ets1','elf1':'elf1','srebf2':'srebf2','ep300':'ep300','pbx3':'pbx3','creb1':'creb1','rfx5':'rfx5','irf4':'irf4','nfic':'nfic','elk1':'elk1','tcf12':'tcf12','nfyb':'nfyb','wrnip1':'wrnip1','zbtb33':'zbtb33','nfya':'nfya','cux1':'cux1','nfatc1':'nfatc1','ctcf':'ctcf','kat2b':'kat2b','batf':'batf','srf':'srf','ezh2':'ezh2'}



num_layers = 2
num_attn_heads = 4
label_size = len(data['dict']['tgt'])-4
global_dec_self_attns = torch.zeros(num_layers,num_attn_heads,label_size,label_size).cuda()




def get_average_attn_weights(data_object):
	# cmap = mpl.cm.bwr
	# cmap = mpl.colors.LinearSegmentedColormap.from_list('mycmap', ['black', 'lime'])

	# cmap = mpl.cm.viridis
	
	norm = mpl.colors.Normalize(vmin=0, vmax=1)
	
	Flag = False
	idx = 0
	
	model = Transformer(
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



	if torch.cuda.device_count() > 1 and opt.multi_gpu:
		print("Using", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)
	if torch.cuda.is_available() and opt.cuda:
		model = model.cuda()
		if opt.gpu_id != -1:
			torch.cuda.set_device(opt.gpu_id)

	
	# state_dict = torch.load(opt.results_dir+'/'+opt.mname+'/model.chkpt')
	state_dict = torch.load(opt.model_name+'/model.chkpt')
	
	model.load_state_dict(state_dict['model'])

	model.eval()


	label_list = []
	for key,value in data['dict']['tgt'].items(): 
		if value >= 4:
			label_list.append(key)

	sorted_labels = sorted(data['dict']['tgt'].items(), key=operator.itemgetter(1))


	rev_tgt_dict = {}
	for key,value in data['dict']['tgt'].items(): 
		rev_tgt_dict[value] = key

	
	rev_src_dict = {}
	for key,value in data['dict']['src'].items():
		rev_src_dict[value] = key

	def get_labels(tgt):
		ret_array = []
		for label_index in tgt:
			for label, index in data['dict']['tgt'].items():
				if index == label_index:
					ret_array.append(label)
		return ret_array


	def get_input(src):
		ret_array = []
		for label_index in src:
			for label, index in data['dict']['src'].items():
				if index == label_index:
					ret_array.append(label)
		return ret_array


	sample_count = 0
	for batch in tqdm(data_object, mininterval=0.5,desc='(Training)   ', leave=False):
			 
		src,adj,tgt = batch
		gold = tgt[:, 1:]
		
		gold_binary = utils.get_gold_binary(gold.data.cpu(),opt.tgt_vocab_size).cuda()
		
		pred,enc_out,attns1,attns2 = model(src, adj,tgt, gold_binary, return_attns=True,int_preds=False)
		# pred,enc_output,*results = model(src,adj,tgt,gold_binary,int_preds=opt.int_preds)

		_,_,all_int_preds = model(src, adj,None, gold_binary, return_attns=False,int_preds=True)
		
		enc_self_attns_layers = attns1[0]
		dec_self_attns_layers = attns2[0]
		enc_dec_attn_layers = attns2[1]

		src = src[0].data
		tgt = tgt.data

			
		for sample_idx in range(src.size(0)):
			sample_count+=1


			src_i = src[sample_idx][src[sample_idx].nonzero()[:,0]][1:-1]
			tgt_i = tgt[sample_idx][tgt[sample_idx].nonzero()[:,0]][1:-1]
			

			pred_i = pred[sample_idx]
			pred_i = F.sigmoid(pred_i).cpu().data

			gold = torch.zeros(pred_i.size(0)).index_fill_(0,tgt_i.cpu()-4,1)

			# inputs = get_input(src_i)
			labels = get_labels(tgt_i)

			if len(labels)>3 and sample_count>50:

				word_list = []
				for pos1 in range(src_i.size(0)): 
					word_list.append(rev_src_dict[src_i[pos1].item()])

				pos_labels = []
				pos_labels_tensor = []
				sample_label_list = []
				for key,value in sorted_labels: 
					if value >= 4:
						if key in labels:
							pos_labels.append(value-4)
							# pos_labels_tensor.append(value)
						sample_label_list.append(key)
				pos_labels_tensor = torch.Tensor(pos_labels).long()

				
				enc_dec_self_attns = torch.zeros(len(enc_dec_attn_layers),num_attn_heads,enc_dec_attn_layers[0].size(1),src_i.size(0)).cuda()
				curr_dec_self_attns = torch.zeros(len(enc_dec_attn_layers),num_attn_heads,dec_self_attns_layers[0].size(1),dec_self_attns_layers[0].size(1)).cuda()
				for layer in range(len(dec_self_attns_layers)):
					enc_self_attns = enc_self_attns_layers[layer].data
					dec_self_attns = dec_self_attns_layers[layer].data 
					enc_dec_attns = enc_dec_attn_layers[layer].data
					
					enc_self_attns = enc_self_attns.view(opt.n_head,-1,enc_self_attns.size(1),enc_self_attns.size(2))
					dec_self_attns = dec_self_attns.view(opt.n_head,-1,dec_self_attns.size(1),dec_self_attns.size(2))
					enc_dec_attns = enc_dec_attns.view(opt.n_head,-1,enc_dec_attns.size(1),enc_dec_attns.size(2))
					# stop()
					
					if get_enc_self_attns:
						enc_self_attns_i = enc_self_attns[sample_idx][:,1:-1,1:-1]
						enc_self_attns_i = enc_self_attns_i[:,0:src_i.size(0),0:src_i.size(0)]
						enc_self_attns_i = torch.mean(enc_self_attns_i,0) # Mean across K heads
					if get_enc_dec_attns:
						# stop()
						enc_dec_attns_i = enc_dec_attns[:,sample_idx,:,:]
						enc_dec_attns_i = enc_dec_attns_i[:,:,1:src_i.size(0)+1]
						enc_dec_self_attns[layer] = enc_dec_attns_i
					if get_dec_self_attns:
						dec_self_attns_i = dec_self_attns[:,sample_idx,:,:]

						# undirected_dec_self_attns_i = (dec_self_attns_i + dec_self_attns_i.transpose(1,2))/2
						undirected_dec_self_attns_i = dec_self_attns_i
						global_dec_self_attns[layer] += undirected_dec_self_attns_i
						curr_dec_self_attns[layer] = undirected_dec_self_attns_i

				
					
				


				enc_dec_self_attns = torch.index_select(enc_dec_self_attns,2,pos_labels_tensor.cuda())

				
				################## CLUSTERING ################

				for layer in range(enc_dec_self_attns.size(0)):
					print(layer)
					mean_attns = enc_dec_self_attns[layer].mean(0).cpu().numpy()
					
					cluster_model = cluster.bicluster.SpectralBiclustering(n_clusters=4, method='log',random_state=0)
					cluster_model.fit(mean_attns)
					fit_data = mean_attns[np.argsort(cluster_model.row_labels_)]
					fit_data = fit_data[:, np.argsort(cluster_model.column_labels_)]
					
					word_list_np = np.array(word_list)
					labels_np = np.array(labels)
					row_indices = np.argsort(cluster_model.row_labels_)
					column_indices = np.argsort(cluster_model.column_labels_)
					word_list_np[row_indices]

					word_list_sorted = word_list_np[column_indices].tolist()
					labels_sorted = labels_np[row_indices].tolist()


					# fig = plt.figure()
					# ax = fig.add_subplot(111)
					# cax = ax.matshow(fit_data,cmap=plt.cm.Blues)
					# ax.set_xticks(np.arange(0,len(word_list_sorted),1))
					# ax.set_xticklabels(word_list_sorted, fontsize=2)
					# plt.setp(ax.get_xticklabels(), rotation=90)
					# ax.set_yticks(np.arange(0,len(labels_sorted),1))
					# ax.set_yticklabels(labels_sorted, fontsize=2)
					# # plt.title("After biclustering; rearranged to show biclusters")
					# plt.savefig('enc_dec_clustering'+str(layer)+'_1.png',dpi=500)


					fig = plt.figure()
					ax = fig.add_subplot(111)
					cax = ax.matshow(np.outer(np.sort(cluster_model.row_labels_) + 1,np.sort(cluster_model.column_labels_) + 1),cmap=plt.cm.Blues)
					ax.set_xticks(np.arange(0,len(word_list_sorted),1))
					ax.set_xticklabels(word_list_sorted, fontsize=2)
					plt.setp(ax.get_xticklabels(), rotation=90)
					ax.set_yticks(np.arange(0,len(labels_sorted),1))
					ax.set_yticklabels(labels_sorted, fontsize=2)
					# plt.title("Checkerboard structure of rearranged data")
					plt.savefig('enc_dec_clustering'+str(layer)+'_2.png',dpi=500)

					##############################

					mean_attns = curr_dec_self_attns[layer].mean(0).cpu().numpy()
					
					cluster_model = cluster.bicluster.SpectralBiclustering(n_clusters=4, method='log',random_state=0)
					cluster_model.fit(mean_attns)
					fit_data = mean_attns[np.argsort(cluster_model.row_labels_)]
					fit_data = fit_data[:, np.argsort(cluster_model.column_labels_)]
					

					row_indices = np.argsort(cluster_model.row_labels_)
					column_indices = np.argsort(cluster_model.column_labels_)

					full_labels = np.arange(4,len(data['dict']['tgt']),1)
					full_labels = get_labels(full_labels)
					labels_np = np.array(full_labels)
					labels_sorted_row = labels_np[row_indices].tolist()
					labels_sorted_column = labels_np[column_indices].tolist()


					# fig = plt.figure()
					# ax = fig.add_subplot(111)
					# cax = ax.matshow(fit_data,cmap=plt.cm.Blues)
					# ax.set_xticks(np.arange(0,len(labels_sorted_column),1))
					# ax.set_xticklabels(labels_sorted_column, fontsize=2)
					# plt.setp(ax.get_xticklabels(), rotation=90)
					# ax.set_yticks(np.arange(0,len(labels_sorted_row),1))
					# ax.set_yticklabels(labels_sorted_row, fontsize=4)
					# # plt.title("After biclustering; rearranged to show biclusters")
					# plt.savefig('dec_clustering'+str(layer)+'.1.png',dpi=500)


					fig = plt.figure()
					ax = fig.add_subplot(111)
					cax = ax.matshow(np.outer(np.sort(cluster_model.row_labels_) + 1,np.sort(cluster_model.column_labels_) + 1),cmap=plt.cm.Blues)
					ax.set_xticks(np.arange(0,len(labels_sorted_column),1))
					ax.set_xticklabels(labels_sorted_column, fontsize=2)
					plt.setp(ax.get_xticklabels(), rotation=90)
					ax.set_yticks(np.arange(0,len(labels_sorted_row),1))
					ax.set_yticklabels(labels_sorted_row, fontsize=2)
					# plt.title("Checkerboard structure of rearranged data")
					plt.savefig('dec_clustering'+str(layer)+'.2.png',dpi=500)
				stop()


					


				enc_dec_self_attns = torch.sum(enc_dec_self_attns,1).unsqueeze(1)
				for layer in range(enc_dec_self_attns.size(0)):
					for head in range(enc_dec_self_attns[layer].size(0)):

						adj_matrix = enc_dec_self_attns[layer][head].cpu().numpy()
						
						fig, ax = plt.subplots()

						# for edge, spine in ax.spines.items():spine.set_visible(False)
						cmap = mpl.cm.Blues
						im = ax.imshow(adj_matrix, cmap=cmap,norm= mpl.colors.Normalize(vmin=0, vmax=1))
						ax.set_xticks(np.arange(0,adj_matrix.shape[1],1))
						ax.set_yticks(np.arange(0,adj_matrix.shape[0],1))
						ax.set_xticklabels(word_list, fontsize=8)
						# ax.set_yticklabels(label_list, fontsize=3)
						# for pos_label in pos_labels: ax.get_yticklabels()[pos_label].set_color("red")
						ax.set_yticklabels(labels, fontsize=8)
						plt.tick_params(axis='y',which='both',bottom=False,top=False,labelbottom=False,pad=0,length=0,colors='red')
						plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True,pad=0,length=0)
						ax.tick_params(axis='both', which='major', pad=1)
						plt.setp(ax.get_xticklabels(), rotation=90)
						cbar = fig.colorbar(im, ticks=[0, 0.5, 1])
						cbar.ax.set_yticklabels(['0', '0.5', '1'])
						plot_file_name = path.join(opt.save_dir,'sample'+str(sample_count)+'.encdec.layer'+str(layer+1)+'.head'+str(head+1)+'.png')
						print(plot_file_name)
						plt.savefig(plot_file_name,dpi=500)#,bbox_inches = 'tight')


				curr_dec_self_attns = torch.index_select(curr_dec_self_attns,2,pos_labels_tensor.cuda())
				curr_dec_self_attns = torch.index_select(curr_dec_self_attns,3,pos_labels_tensor.cuda())


				curr_dec_self_attns = torch.sum(curr_dec_self_attns,1).unsqueeze(1)
				
				for layer in range(curr_dec_self_attns.size(0)):
					for head in range(curr_dec_self_attns[layer].size(0)):
						# mean_global = (global_dec_self_attns[layer][head] + global_dec_self_attns[layer][head].transpose(0,1))/2.
						# adj_matrix = mean_global.cpu().numpy()
						adj_matrix = curr_dec_self_attns[layer][head].cpu().numpy()# adj_matrix[adj_matrix == 0] = 'nan'
						if opt.normalize:
							max_vals = np.nanmax(adj_matrix, axis=1)
							min_vals = np.nanmin(adj_matrix, axis=1)
							max_vals = np.repeat(max_vals.reshape((-1,1)),adj_matrix.shape[1],1)
							min_vals = np.repeat(min_vals.reshape((-1,1)),adj_matrix.shape[1],1)
							adj_matrix = (adj_matrix-min_vals)/(max_vals-min_vals)

						if not opt.one_sample:
							adj_matrix = adj_matrix/sample_count
							adj_matrix = preprocessing.minmax_scale(adj_matrix,feature_range=(0,1))
						
						fig, ax = plt.subplots()
						cmap = mpl.cm.Blues
						im = ax.imshow(adj_matrix, cmap=cmap,norm= mpl.colors.Normalize(vmin=0, vmax=1))
						ax.set_xticks(np.arange(adj_matrix.shape[0]))
						ax.set_yticks(np.arange(adj_matrix.shape[1]))
						# ax.set_xticklabels(label_list, fontsize=3)
						# ax.set_yticklabels(label_list, fontsize=3)
						plt.xlabel('Input Features')
						plt.ylabel('Labels')
						ax.set_xticklabels(labels, fontsize=40)
						ax.set_yticklabels(labels, fontsize=40)
						plt.tick_params(axis='y',which='both',bottom=False,top=False,labelbottom=False,pad=2,length=0,colors='red')
						plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True,pad=2,length=0,colors='red')
						ax.tick_params(axis='both', which='major', pad=2)
						plt.setp(ax.get_xticklabels(), rotation=90)
						cbar = fig.colorbar(im, ticks=[0, 0.5, 1])
						cbar.ax.set_yticklabels(['0', '0.5', '1'])
						# ax.set_title("adj matrix of head i")
						plot_file_name = path.join(opt.save_dir,'sample'+str(sample_count)+'.decself.layer'+str(layer+1)+'.head'+str(head+1)+'.png')
						print(plot_file_name)
						plt.savefig(plot_file_name,dpi=500,bbox_inches = 'tight')


				all_preds = torch.Tensor()
			
				for int_pred in all_int_preds:
					int_pred = F.sigmoid(int_pred[sample_idx]).cpu().data
					all_preds = torch.cat((all_preds,int_pred.unsqueeze(0)),0)
				all_preds = torch.cat((all_preds,pred_i.unsqueeze(0)),0)

				all_preds = torch.flip(all_preds,[0])
				
				sample_label_list = ['']
				pos_labels = []
				for key,value in sorted_labels: 
					if value >= 4:
						if key in labels:
							pos_labels.append(value-4+1)
						sample_label_list.append(key)

				adj_matrix = all_preds.numpy()
				

				fig, ax = plt.subplots()
				cmap = mpl.cm.viridis
				im = ax.imshow(adj_matrix,cmap=cmap,norm= mpl.colors.Normalize(vmin=0, vmax=1))
				ax.set_xticks(np.arange(-.5, len(sample_label_list),1))
				ax.set_yticks(np.arange(0, adj_matrix.shape[0],1))
				ax.set_yticklabels(['Step 2.2: Label-Label MP','Step 2.1: Feature-Label MP','Step 1.2: Label-Label MP','Step 1.1: Feature-Label MP'], fontsize=1.75)
				ax.set_xticklabels(sample_label_list, fontsize=1.75)
				plt.tick_params(axis='y',which='both',bottom=False,top=False,labelbottom=False,pad=1.0,length=0)
				plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True,pad=0.5,length=0)
				plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
				ax.grid(linestyle='-', linewidth='0.0', color='w')

				plt.xlabel('Labels')
				plt.xlabel('Label Node Update Steps')
				
				cbar.ax.set_yticklabels(['0', '0.5', '1']) 
				divider = make_axes_locatable(ax)
				cax = divider.append_axes("right", size="5%", pad=0.05)
				cbar = fig.colorbar(im, ticks=[0, 0.5, 1],cax=cax)

				for pos_label in pos_labels: ax.get_xticklabels()[pos_label].set_color("red")
				plot_file_name = path.join(opt.save_dir,'sample'+str(sample_count)+'.int_preds.png')
				print(plot_file_name)
				fig.tight_layout()
				plt.savefig(plot_file_name,dpi=1000)



				pos_labels_predictions = torch.index_select(all_preds,1,pos_labels_tensor)
				adj_matrix = pos_labels_predictions.numpy()
			
				fig, ax = plt.subplots()
				cmap = mpl.cm.viridis
				im = ax.imshow(adj_matrix,cmap=cmap,norm= mpl.colors.Normalize(vmin=0, vmax=1))
				ax.set_xticks(np.arange(0.5, adj_matrix.shape[1],1))
				ax.set_yticks(np.arange(0.5, adj_matrix.shape[0],1))
				ax.set_yticklabels(['label to label out 2','label to input out 2','label to label out 1','label to input out 1'], fontsize=25)
				ax.set_xticklabels(labels, fontsize=30)
				plt.tick_params(axis='y',which='both',bottom=False,top=False,labelbottom=False,pad=0.5,length=0)
				plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True,pad=0.5,length=0)
				plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
				ax.grid(linestyle='-', linewidth='0.3', color='w')
				cbar.ax.set_yticklabels(['0', '0.5', '1']) 
				divider = make_axes_locatable(ax)
				cax = divider.append_axes("right", size="5%", pad=0.05)
				cbar = fig.colorbar(im, ticks=[0, 0.5, 1],cax=cax)
				plot_file_name = path.join(opt.save_dir,'sample'+str(sample_count)+'.int_preds2.png')
				print(plot_file_name)
				fig.tight_layout()
				plt.savefig(plot_file_name,dpi=500)

	
	montage = False
	if montage:
		if opt.normalize:
			montage_file_name = opt.save_dir+'/'+opt.data_type+'_norm.pdf'
		else:
			montage_file_name = opt.save_dir+'/'+opt.data_type+'.pdf'
		print('montage '+opt.save_dir+'/*.png -tile 4x2 -geometry +2+2 '+montage_file_name)
		os.system('montage '+opt.save_dir+'/*.png -tile 4x2 -geometry +2+2 '+montage_file_name)
	


def get_int_preds(data_object):
	Flag = False
	idx = 0
	
	model = Transformer(
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



	if torch.cuda.device_count() > 1 and opt.multi_gpu:
		print("Using", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)
	if torch.cuda.is_available() and opt.cuda:
		model = model.cuda()
		if opt.gpu_id != -1:
			torch.cuda.set_device(opt.gpu_id)

	
	# state_dict = torch.load(opt.results_dir+'/'+opt.mname+'/model.chkpt')
	state_dict = torch.load(opt.model_name+'/model.chkpt')
	
	model.load_state_dict(state_dict['model'])

	model.eval()


	label_list = []
	for key,value in data['dict']['tgt'].items(): 
		if value >= 4:
			label_list.append(key)

	sorted_labels = sorted(data['dict']['tgt'].items(), key=operator.itemgetter(1))


	rev_tgt_dict = {}
	for key,value in data['dict']['tgt'].items(): 
		rev_tgt_dict[value] = key

	rev_src_dict = {}
	for key,value in data['dict']['src'].items():
		rev_src_dict[value] = key


	def get_labels(tgt):
		ret_array = []
		for label_index in tgt:
			for label, index in data['dict']['tgt'].items():
				if index == label_index:
					ret_array.append(label)
		return ret_array


	def get_input(src):
		ret_array = []
		for label_index in src:
			for label, index in data['dict']['src'].items():
				if index == label_index:
					ret_array.append(label)
		return ret_array


	cmap = mpl.cm.viridis
	# cmap = mpl.colors.LinearSegmentedColormap.from_list('mycmap', ['white','red'])
	norm = mpl.colors.Normalize(vmin=0, vmax=1)

	for batch in tqdm(data_object, mininterval=0.5,desc='(Training)   ', leave=False):
			
		
		src,adj,tgt = batch
		gold = tgt[:, 1:]
		
		gold_binary = utils.get_gold_binary(gold.data.cpu(),opt.tgt_vocab_size).cuda()
		
		pred,enc_out,all_int_preds = model(src, adj,None, gold_binary, return_attns=False,int_preds=True)

		
		src = src[0].data
		tgt = tgt.data

		
		idx+=1
		sample_idx = 0

		src_i = src[sample_idx]
		tgt_i = tgt[sample_idx]

		src_i = src_i[src_i.nonzero()[:,0]][1:-1]
		tgt_i = tgt_i[tgt_i.nonzero()[:,0]][1:-1]

		pred_i = pred[sample_idx]
		pred_i = F.sigmoid(pred_i).cpu().data


		gold = torch.zeros(pred_i.size(0)).index_fill_(0,tgt_i.cpu()-4,1)

		labels = get_labels(tgt_i) # inputs = get_input(src_i)

		all_preds = torch.Tensor()
		

		for int_pred in all_int_preds:
			int_pred = F.sigmoid(int_pred[sample_idx]).cpu().data
			all_preds = torch.cat((all_preds,int_pred.unsqueeze(0)),0)
		all_preds = torch.cat((all_preds,pred_i.unsqueeze(0)),0)

		all_preds = torch.flip(all_preds,[0])

		
		sample_label_list = ['']
		print(labels)
		pos_labels = []
		for key,value in sorted_labels: 
			if value >= 4:
				if key in labels:
					pos_labels.append(value-4+1)
				sample_label_list.append(key)

		print('saving plots')
		adj_matrix = all_preds.numpy()

		# stop()
		fig, ax = plt.subplots()
		for edge, spine in ax.spines.items():spine.set_visible(False)
		im = ax.imshow(adj_matrix,cmap=cmap,norm=norm)
		ax.set_xticks(np.arange(-.5, len(sample_label_list),1))
		ax.set_yticks(np.arange(0, len(all_preds),1))
		ax.set_yticklabels(['label to label out 2','label to input out 2','label to label out 1','label to input out 1'], fontsize=3)
		ax.set_xticklabels(sample_label_list, fontsize=3)
		plt.tick_params(axis='y',which='both',bottom=False,top=False,labelbottom=False,pad=0,length=0)
		plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True,pad=0,length=0)
		plt.setp(ax.get_xticklabels(), rotation=90, ha="right")#,rotation_mode="anchor")
		for pos_label in pos_labels: ax.get_xticklabels()[pos_label].set_color("red")
		plot_file_name = path.join(opt.save_dir,'sample'+str(sample_idx+1)+'.int_preds.png')
		print(plot_file_name)
		fig.tight_layout()
		plt.savefig(plot_file_name,dpi=500)#,bbox_inches = 'tight')
		 # ax.grid(linestyle='-', linewidth='0.3', color='w')


		if opt.one_sample:
			break





def get_uncoditional_dependence(data):

	label_list = []
	for key,value in data['dict']['tgt'].items(): 
		if value >= 4:
			label_list.append(key)
	
	cmap = mpl.cm.bwr

	cdict = {'green':   ((0.0, 0.0, 0.0),
				   (0.5, 0.0, 0.1),
				   (1.0, 1.0, 1.0)),
		 'blue': ((0.0, 0.0, 0.0),
				   (1.0, 0.0, 0.0)),
		 'red':  ((0.0, 0.0, 1.0),
				   (0.5, 0.1, 0.0),
				   (1.0, 0.0, 0.0))
		}
	# cmap = mpl.colors.LinearSegmentedColormap('green_red', cdict,N=256)
	cmap = mpl.colors.LinearSegmentedColormap.from_list('mycmap', ['red', 'black', 'lime'])
	norm = mpl.colors.Normalize(vmin=-1, vmax=1)



	train_label_vals = torch.zeros(len(data['train']['tgt']),len(data['dict']['tgt']))
	for i in range(len(data['train']['tgt'])):
		indices = torch.from_numpy(np.array(data['train']['tgt'][i]))
		x = torch.zeros(len(data['dict']['tgt']))
		x.index_fill_(0, indices, 1)
		train_label_vals[i] = x

	train_label_vals = train_label_vals[:,4:]

	co_occurence_matrix = torch.zeros(train_label_vals.size(1),train_label_vals.size(1))
	for sample in data['train']['tgt']:
		sample2 = sample
		for i,idx1 in enumerate(sample[1:-1]):
			for idx2 in sample2[i+1:-1]:
				if idx1 != idx2:
					co_occurence_matrix[idx1-4,idx2-4] += 1


	print(co_occurence_matrix.sum().item())

	co_occurence_matrix[co_occurence_matrix>0]=1

	print(co_occurence_matrix.sum().item())

	pearson_matrix = np.corrcoef(train_label_vals.transpose(0,1).cpu().numpy())


	adj_matrix = pearson_matrix
	adj_matrix[adj_matrix >= 0.9999] = 'nan'
	adj_matrix2 = np.nan_to_num(adj_matrix)


	print(abs(adj_matrix2).sum().item())

	adj_matrix2[adj_matrix2 < 0] = 0
	print(adj_matrix2.sum().item())


	adj_matrix = pearson_matrix
	adj_matrix[adj_matrix >= 0.9999] = 'nan'
	adj_matrix2 = np.nan_to_num(adj_matrix)
	
	adj_matrix2[adj_matrix2 < 0.1] = 0
	adj_matrix2[adj_matrix2 > 0] = 1
	print(adj_matrix2.sum().item())


	fig, ax = plt.subplots()
	im = ax.imshow(adj_matrix, cmap=cmap,norm=norm)
	ax.set_xticks(np.arange(len(label_list)))
	ax.set_yticks(np.arange(len(label_list)))
	ax.set_xticklabels(label_list, fontsize=3)
	ax.set_yticklabels(label_list, fontsize=3)
	ax.tick_params(axis='both', which='major', pad=1)
	plt.setp(ax.get_xticklabels(), rotation=90)
			
	fig.colorbar(im)
	plot_file_name = path.join(opt.save_dir,'unconditional_dependencies.png')
	plt.savefig(plot_file_name,dpi=500,bbox_inches = 'tight')





	




def main():
	get_average_attn_weights(data_object)
	# get_int_preds(data_object)
	# get_uncoditional_dependence(data)
  
if __name__== "__main__":
  main()

