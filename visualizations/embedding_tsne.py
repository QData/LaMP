
import argparse
import math
import time
from pdb import set_trace as stop
import numpy
import numpy as np
import warnings
import os.path as path 


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable







from sklearn.manifold import TSNE
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

parser.add_argument('-dataroot', type=str, default='data/')
parser.add_argument('-dataset', type=str, default='rcv1')
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-dropout', type=float, default=0.2)
parser.add_argument('-dec_dropout', type=float, default=0.2)
parser.add_argument('-d_model', type=int, default=512)
parser.add_argument('-d_inner_hid', type=int, default=1024)
parser.add_argument('-n_head', type=int, default=4)
parser.add_argument('-n_layers_enc', type=int, default=3)
parser.add_argument('-n_layers_dec', type=int, default=3)
parser.add_argument('-encoder', type=str, default='sa')
parser.add_argument('-decoder', type=str, default='pmlp')
parser.add_argument('-enc_transform', type=str, default='mean')
parser.add_argument('-onehot', action='store_true')
parser.add_argument('-no_enc_pos_embedding', action='store_true')
parser.add_argument('-dec_reverse', action='store_true')
parser.add_argument('-no_residual', action='store_true')
parser.add_argument('-name', type=str, default=None)
opt = parser.parse_args()

opt.cuda = True

opt.results_dir = path.join('results',opt.dataset)

opt.dataset = path.join(opt.dataroot,opt.dataset)

opt.data = path.join(opt.dataset,'train_valid_test.pt')


opt.mname='enc_sa.dec_pmlp.512.1024.128.128.nlayers_3_3.nheads_4.proj_share.bsz_32.adam.lr_0002.dropout_20_20.threshold_40.no_zero_pad'









rcv1_dict = {'ccat':'corporate/industrial','c11':'strategy/plans','c12':'legal/judicial','c13':'regulation/policy','c14':'share listings','c15':'performance','c151':'accounts/earnings','c1511':'annual results','c152':'comment/forecasts','c16':'insolvency/liquidity','c17':'funding/capital','c171':'share capital','c172':'bonds/debt issues','c173':'loans/credits','c174':'credit ratings','c18':'ownership changes','c181':'mergers/acquisitions','c182':'asset transfers','c183':'privatisations','c21':'production/services','c22':'new products/services','c23':'research/development','c24':'capacity/facilities','c31':'markets/marketing','c311':'domestic markets','c312':'external markets','c313':'market share','c32':'advertising/promotion','c33':'contracts/orders','c331':'defence contracts','c34':'monopolies/competition','c41':'management','c411':'management moves','c42':'labour','ecat':'economics','e11':'economic performance','e12':'monetary/economic','e121':'money supply','e13':'inflation/prices','e131':'consumer prices','e132':'wholesale prices','e14':'consumer finance','e141':'personal income','e142':'consumer credit','e143':'retail sales','e21':'government finance','e211':'expenditure/revenue','e212':'government borrowing','e31':'output/capacity','e311':'industrial production','e312':'capacity utilization','e313':'inventories','e41':'employment/labour','e411':'unemployment','e51':'trade/reserves','e511':'balance of payments','e512':'merchandise trade','e513':'reserves','e61':'housing starts','e71':'leading indicators','gcat':'government/social','g15':'european community','g151':'ec internal market','g152':'ec corporate policy','g153':'ec agriculture policy','g154':'ec monetary/economic','g155':'ec institutions','g156':'ec environment issues','g157':'ec competition/subsidy','g158':'ec external relations','g159':'ec general','gcrim':'crime/law enforcement','gdef':'defence','gdip':'international relations','gdis':'disasters and accidents','gent':'arts/culture/entertainment','genv':'environment and natural world','gfas':'fashion','ghea':'health','gjob':'labour issues','gmil':'millennium issues','gobit':'obituaries','godd':'human interest','gpol':'domestic politics','gpro':'biographies/personalities/people','grel':'religion','gsci':'science and technology','gspo':'sports','gtour':'travel and tourism','gvio':'war/civil war','gvote':'elections','gwea':'weather','gwelf':'welfare/social services','mcat':'markets','g15':'european community','gcrim':'crime/law enforcement','gdef':'defence','gdip':'international relations','gdis':'disasters and accidents','gent':'arts/culture/entertainment','genv':'environment and natural world','gfas':'fashion','ghea':'health','gjob':'labour issues','gmil':'millennium issues','gobit':'obituaries','godd':'human interest','gpol':'domestic politics','gpro':'biographies/personalities/people','grel':'religion','gsci':'science and technology','gspo':'sports','gtour':'travel and tourism','gvio':'war/civil war','gvote':'elections','gwea':'weather','gwelf':'welfare/social services','m11':'equity markets','m12':'bond markets','m13':'money markets','m131':'interbank markets','m132':'forex markets','m14':'commodity markets','m141':'soft commodities','m142':'metals trading','m143':'energy markets'}

categories = {1 : ['ccat','c11','c12','c13','c14','c15','c151','c1511','c152','c16','c17','c171','c172','c173','c174','c18','c181','c182','c183','c21','c22','c23','c24','c31','c311','c312','c313','c32','c33','c331','c34','c41','c411','c42'],2 : ['ecat','e11','e12','e121','e13','e131','e132','e14','e141','e142','e143','e21','e211','e212','e31','e311','e312','e313','e41','e411','e51','e511','e512','e513','e61','e71'], 3 : ['gcat','g15','g151','g152','g153','g154','g155','g156','g157','g158','g159','gcrim','gdef','gdip','gdis','gent','genv','gfas','ghea','gjob','gmil','gobit','godd','gpol','gpro','grel','gsci','gspo','gtour','gvio','gvote','gwea','gwelf'], 4 : ['mcat','g15','gcrim','gdef','gdip','gdis','gent','genv','gfas','ghea','gjob','gmil','gobit','godd','gpol','gpro','grel','gsci','gspo','gtour','gvio','gvote','gwea','gwelf','m11','m12','m13','m131','m132','m14','m141','m142','m143']}




def get_embedding():
	
	import transformer.Constants as Constants
	from transformer.Models import Transformer
	from transformer.Optim import ScheduledOptim
	from transformer.Modules import LabelSmoothing
	from transformer.Beam import Beam
	from transformer.Translator import translate
	from preprocess import read_instances_from_file, convert_instance_to_idx_seq
	import evals
	from evals import Logger
	from DataLoader import DataLoader


	data = torch.load(opt.data)

	opt.max_token_seq_len_e = data['settings'].max_seq_len
	opt.max_token_seq_len_d = 30
	opt.proj_share_weight = True
	opt.d_word_vec = opt.d_model

	# training_data = DataLoader(
 #    data['dict']['src'],
 #    data['dict']['tgt'],
 #    src_insts=data['train']['src'],
 #    tgt_insts=data['train']['tgt'],
 #    batch_size=opt.batch_size,
 #    shuffle=True,
 #    cuda=opt.cuda)


	opt.src_vocab_size = training_data.src_vocab_size
	opt.tgt_vocab_size = training_data.tgt_vocab_size
	opt.tgt_vocab_size = opt.tgt_vocab_size - 4


	opt.src_vocab_size = training_data.src_vocab_size
	opt.tgt_vocab_size = training_data.tgt_vocab_size
	opt.tgt_vocab_size = opt.tgt_vocab_size - 4



	opt.d_v = int(opt.d_model/opt.n_head)
	opt.d_k = int(opt.d_model/opt.n_head)

	model = Transformer(
	        opt.src_vocab_size,
	        opt.tgt_vocab_size,
	        opt.max_token_seq_len_e,
	        opt.max_token_seq_len_d,
	        proj_share_weight=opt.proj_share_weight,
	        embs_share_weight=False,
	        d_k=opt.d_k,
	        d_v=opt.d_v,
	        d_model=opt.d_model,
	        d_word_vec=opt.d_word_vec,
	        d_inner_hid=opt.d_inner_hid,
	        n_layers_enc=opt.n_layers_enc,
	        n_layers_dec=opt.n_layers_dec,
	        n_head=opt.n_head,
	        dropout=opt.dropout,
	        dec_dropout=opt.dec_dropout,
	        encoder=opt.encoder,
	        decoder=opt.decoder,
	        enc_transform=opt.enc_transform,
	        onehot=opt.onehot,
	        no_enc_pos_embedding=opt.no_enc_pos_embedding,
	        dec_reverse=opt.dec_reverse,
	        no_residual=opt.no_residual)



	state_dict = torch.load(opt.results_dir+'/'+opt.mname+'/model.chkpt')


	model.load_state_dict(state_dict['model'])

	model = model.cuda()
	model.eval()


	model.decoder.tgt_word_emb.weight



	W = model.decoder.tgt_word_emb.weight.data.cpu().numpy()


	numpy.save(W,'Embedding')


def visualize_embedding():
	
	import matplotlib
	matplotlib.use('agg')

	from matplotlib import pyplot as plt

	label_dict = torch.load('label_dict.pt')

	rev_label_dict = {}
	for key,value in label_dict.items(): 
		rev_label_dict[value] = key

	W = numpy.load('Embedding.npy')

	colors = []
	for i in range(len(W)):
		label = rev_label_dict[i+4]
		if label in categories[1]:
			colors.append(1)
		elif label in categories[2]:
			colors.append(2)
		elif label in categories[3]:
			colors.append(3)
		elif label in categories[4]:
			colors.append(4)
		else:
			print(label)

	colors = np.array(colors)
	colors = colors.astype(float)


	W_2 = TSNE(n_components=2, perplexity=10.0, early_exaggeration=10.0, learning_rate=5, n_iter=10000, n_iter_without_progress=1000, min_grad_norm=1e-07, metric='euclidean').fit_transform(W)
	plt.scatter(W_2[:,0],W_2[:,1],s=3,c=colors)
	plt.savefig('tsne.png',dpi = 800)
	plt.close()
	# plt.show()



# get_embedding()
visualize_embedding()






