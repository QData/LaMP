#!/usr/bin/env python
import argparse
from pdb import set_trace as stop
import os.path as path 
import sys
import csv
from pathlib import Path


parser = argparse.ArgumentParser()

parser.add_argument('-decoder', type=str, default='mlp')
parser.add_argument('-dataset', type=str, default='reuters') 
parser.add_argument('-max', type=int, default=100) 

opt = parser.parse_args()



if opt.dataset in ['nuswide_vector','gene','scene','yeast','tcell','tcell2']:
	root = 'results'
else:
	root = '/bigtemp/jjl5sw/deepENCODE/results/'
	




variations = {

# 'mean_20_20':[
# 'enc_emb.et_sum.dec_graph.512.512.128.128.nlayers_1_1.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.inveyemask',
# 'enc_emb.et_sum.dec_graph.512.512.128.128.nlayers_1_1.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.nonemask',
# 'enc_emb.et_sum.dec_graph.512.512.128.128.nlayers_1_1.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.priormask',
# ],

# 'mean_10_10':[
# 'enc_emb.et_sum.dec_graph.512.512.128.128.nlayers_1_1.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.inveyemask',
# 'enc_emb.et_sum.dec_graph.512.512.128.128.nlayers_1_1.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.nonemask',
# 'enc_emb.et_sum.dec_graph.512.512.128.128.nlayers_1_1.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.priormask',
# ],


# # 'emb_20_20':[
# # 'enc_emb.dec_graph.512.512.128.128.nlayers_1_1.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.inveyemask',
# # 'enc_emb.dec_graph.512.512.128.128.nlayers_1_1.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.nonemask',
# # 'enc_emb.dec_graph.512.512.128.128.nlayers_1_1.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.priormask',
# # ],

# 'emb_20_20':[
# 'enc_emb.dec_graph.512.512.128.128.nlayers_1_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.inveyemask',
# 'enc_emb.dec_graph.512.512.128.128.nlayers_1_1.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.nonemask',
# 'enc_emb.dec_graph.512.512.128.128.nlayers_1_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.nonemask',
# 'enc_emb.dec_graph.512.512.128.128.nlayers_1_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.priormask',
# ],

# 'emb_10_10':[
# 'enc_emb.dec_graph.512.512.128.128.nlayers_1_1.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.inveyemask',
# 'enc_emb.dec_graph.512.512.128.128.nlayers_1_1.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.nonemask',
# 'enc_emb.dec_graph.512.512.128.128.nlayers_1_1.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.priormask',
# ],

'graph_conv':[
'enc_graph.dec_graph.1024.1024.1024.1024.nlayers_1_1.nheads_1.proj_share.bsz_32.loss_ce.adam.lr_0001.drop_10_10.no_dec_self_att',
'enc_graph.dec_graph.1024.1024.1024.1024.nlayers_1_1.nheads_1.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.no_dec_self_att',
'enc_graph.dec_graph.256.256.256.256.nlayers_1_1.nheads_1.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.no_dec_self_att',
'enc_graph.dec_graph.512.512.512.512.nlayers_1_1.nheads_1.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.no_dec_self_att',
'enc_graph.dec_graph.256.256.256.256.nlayers_1_1.nheads_1.proj_share.bsz_32.loss_ce.adam.lr_0005.drop_20_20.no_dec_self_att',
'enc_graph.dec_graph.256.256.256.256.nlayers_1_1.nheads_1.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.no_dec_self_att',
'enc_graph.dec_graph.256.256.256.256.nlayers_1_1.nheads_1.proj_share.bsz_32.loss_ce.adam.lr_0005.drop_10_10.nonemask',
'enc_graph.dec_graph.512.512.512.512.nlayers_1_1.nheads_1.proj_share.bsz_32.loss_ce.adam.lr_0005.drop_10_10.nonemask',
'enc_graph.dec_graph.512.512.512.512.nlayers_1_1.nheads_1.proj_share.bsz_32.loss_ce.adam.lr_0005.drop_20_20.nonemask',
'enc_graph.dec_graph.256.256.256.256.nlayers_1_1.nheads_1.proj_share.bsz_32.loss_ce.adam.lr_0005.drop_20_20.nonemask',
'enc_graph.dec_graph.512.512.512.512.nlayers_1_1.nheads_1.proj_share.bsz_32.loss_ce.adam.lr_0005.drop_10_10.no_dec_self_att',
'enc_graph.dec_graph.256.256.256.256.nlayers_1_1.nheads_1.proj_share.bsz_32.loss_ce.adam.lr_0005.drop_10_10.no_dec_self_att',
'enc_graph.dec_graph.512.512.512.512.nlayers_1_1.nheads_1.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.nonemask',
'enc_graph.dec_graph.512.512.512.512.nlayers_1_1.nheads_1.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.no_dec_self_att',
'enc_graph.dec_graph.512.512.512.512.nlayers_1_1.nheads_1.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.nonemask',
],


# 'graph_20_20':[
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ranking.adam.lr_0002.drop_20_20.nonemask',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ranking.adam.lr_0002.drop_20_20.nonemask.2',
# 'enc_emb.et_mean.dec_mlp.512.512.128.128.nlayers_1_2.nheads_1.bsz_32.loss_ce.adam.lr_0002.drop_20_20',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.inveyemask',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.inveyemask.int_preds_01',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.inveyemask.int_preds_02',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.inveyemask.int_preds_03',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.nonemask',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.nonemask.int_preds_01',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.nonemask.int_preds_02',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.nonemask.int_preds_03',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.priormask',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.priormask.int_preds_01',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.priormask.int_preds_02',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.priormask.int_preds_03',
# # 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.randommask',
# # 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.randommask.int_preds_01',
# # 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.randommask.int_preds_02',
# # 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.randommask.int_preds_03',
# ],


# 'graph_10_10':[
# 'enc_emb.et_mean.dec_mlp.512.512.128.128.nlayers_1_2.nheads_1.bsz_32.loss_ce.adam.lr_0002.drop_10_10',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.inveyemask',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.inveyemask.int_preds_01',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.inveyemask.int_preds_02',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.inveyemask.int_preds_03',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.nonemask',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.nonemask.int_preds_01',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.nonemask.int_preds_02',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.nonemask.int_preds_03',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.priormask',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.priormask.int_preds_01',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.priormask.int_preds_02',
# 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.priormask.int_preds_03',
# # 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.randommask',
# # 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.randommask.int_preds_01',
# # 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.randommask.int_preds_02',
# # 'enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.randommask.int_preds_03',
# ],




}


if opt.dataset in ['nuswide_vector','gene','scene','yeast']:
	variations = {
		'graph_10_10':[
			# 'enc_mlp.dec_mlp.256.256.64.64.nlayers_2_2.nheads_1.bsz_32.loss_ce.adam.lr_0002.drop_10_10',
			# 'enc_mlp.dec_graph.256.256.64.64.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.inveyemask',
			# 'enc_mlp.dec_graph.256.256.64.64.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.nonemask',
			# 'enc_mlp.dec_graph.256.256.64.64.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.nonemask.int_preds_01',
			# 'enc_mlp.dec_graph.256.256.64.64.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.nonemask.int_preds_02',
			# 'enc_mlp.dec_graph.256.256.64.64.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.nonemask.int_preds_03',
			# 'enc_mlp.dec_graph.256.256.64.64.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.priormask',
			# 'enc_mlp.dec_graph.256.256.64.64.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.priormask.int_preds_01',
			# 'enc_mlp.dec_graph.256.256.64.64.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.priormask.int_preds_02',
			# 'enc_mlp.dec_graph.256.256.64.64.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.priormask.int_preds_03',
			'enc_mlp.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.priormask',
			'enc_mlp.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.nonemask',
			'enc_mlp.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.eyemask',
			'enc_mlp.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.nonemask.int_preds_01',
			'enc_mlp.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.priormask.int_preds_01',
			'enc_mlp.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.eyemask.int_preds_01',
			'enc_mlp.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.inveyemask.int_preds_01',
			'enc_mlp.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.nonemask.int_preds_02',
			'enc_mlp.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.priormask.int_preds_02',
			'enc_mlp.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.eyemask.int_preds_02',
			'enc_mlp.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.inveyemask.int_preds_02',
			'enc_mlp.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.nonemask.int_preds_03',
			'enc_mlp.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.priormask.int_preds_03',
			'enc_mlp.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.eyemask.int_preds_03',
			'enc_mlp.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.inveyemask.int_preds_03',
			'enc_mlp.dec_rnn_m.512.512.512.128.nlayers_2_2.nheads_1.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_20_20.ls_10.beam_5',
			'enc_mlp.dec_mlp.512.512.512.128.nlayers_2_2.nheads_1.bsz_32.loss_ce.adam.lr_0002.drop_20_20',
		]
	}

	

if opt.dataset == 'rcv1':
	variations['graph_10_10'] += ['enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.priormask.int_preds_03.undirected']
	variations['graph_10_10'] += ['enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.priormask.int_preds_02.undirected']
	variations['graph_10_10'] += ['enc_graph.dec_graph.512.512.128.128.nlayers_2_2.nheads_4.proj_share.bsz_32.loss_ce.adam.lr_0002.drop_10_10.priormask.int_preds_01.undirected']



THRESHOLDS = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.8,0.85,0.9,0.95]
METRICS = ['ebF1','miF1','maF1','ACC','HA']
#THRESHOLDS=[0.5]


def get_count():
	csv_file = open(path.join('csv_results_files',opt.dataset+'.csv'),'w+')
	csv_file.write('model,count\n')
	for variation,models in variations.items():
		if not ((opt.dataset in ['reuters','bibtext','sider']) and ('10_10' in variation)) and not ((opt.dataset in ['bookmarks','delicious','rcv1','gm12878']) and ('20_20' in variation)):
				print('Variation: '+str(variation))
				csv_file.write(variation+'\n')
				for model in models:
					if (opt.dataset != 'rcv1' and 'matrixlambda' in model) or ((opt.dataset in ['reuters','bibtext']) and ('drop_10_10' in model)):
						pass
					else:
						pred_files_dir = path.join(root,opt.dataset,model,'epochs')
						for i in range(1,101):
							my_file = Path(path.join(pred_files_dir,'test_preds'+str(i)+'.pt'))
							if my_file.is_file():
								pass
							else:
								break
								
						csv_file.write(model)					
						csv_file.write(','+str(i-1))
						csv_file.write('\n')

						if (i-1) < 50:
							sys.stdout.write(model)
							sys.stdout.write(','+str(i-1))
							sys.stdout.write('\n')
				csv_file.write('\n')
				sys.stdout.write('\n')


def precision_at_k(predictions,targets):
	p_value_1 = 0
	p_value_3 = 0
	p_value_5 = 0
	for sample_idx in range(predictions.size(0)):
		
		prediction = predictions[sample_idx].numpy()
		# sorted_predictions = np.sort(prediction)[::-1]
		sorted_indices = np.argsort(prediction)[::-1]


		sorted_targets = targets[sample_idx].numpy().take(sorted_indices)

		p_value_1 += (sorted_targets[0:1].sum()/1)
		p_value_3 += (sorted_targets[0:3].sum()/3)
		p_value_5 += (sorted_targets[0:5].sum()/5)


	p_value_1 = p_value_1/targets.size(0)
	p_value_3 = p_value_3/targets.size(0)
	p_value_5 = p_value_5/targets.size(0)

	# sorted_predictions = np.sort(valid_predictions.numpy(), axis=1)[:,::-1]
	# sorted_indices = np.argsort(valid_predictions.numpy(), axis=1)[:,::-1]

	return p_value_1,p_value_3,p_value_5




def get_results():
	for variation,models in variations.items():
		if not ((opt.dataset in ['reuters','bibtext','sider']) and ('10_10' in variation)) and not ((opt.dataset in ['bookmarks','delicious','rcv1','gm12878']) and ('20_20' in variation)):
				print('====================================================================================')
				print('Variation: '+str(variation))
				csv_file = open(path.join('csv_results_files',opt.dataset+'_'+str(variation)+'.csv'),'w+')

				csv_file.write('model')
				for metric in METRICS:
					csv_file.write(','+metric)
				csv_file.write(',AVG,avgAUC,medAUC,avgAUPR,P@1,P@3,P@5\n')

				for model in models:
					loaded=False

					if (opt.dataset != 'rcv1' and 'matrixlambda' in model):
						pass
					else:
						csv_file.write(model)
						

						

						try:
							best_test_metrics = load(path.join(root,opt.dataset,model+'/final_results.pt'))
							print(path.join(opt.dataset,model))
							print('loaded!')
							loaded=True
							Flag = True
							i = 51
						except:

							try:
								with open(path.join(root,opt.dataset,model,'test_all_auc.csv'), newline='\n') as csvfile:
									test_auc = np.array(list(csv.reader(csvfile))).astype(np.float)
									test_auc = test_auc[:,1:]
								with open(path.join(root,opt.dataset,model,'test_all_aupr.csv'), newline='\n') as csvfile:
									test_aupr = np.array(list(csv.reader(csvfile))).astype(np.float)
									test_aupr = test_aupr[:,1:]
								with open(path.join(root,opt.dataset,model,'valid_all_auc.csv'), newline='\n') as csvfile:
									valid_auc = np.array(list(csv.reader(csvfile))).astype(np.float)
									valid_auc = valid_auc[:,1:]
								with open(path.join(root,opt.dataset,model,'valid_all_aupr.csv'), newline='\n') as csvfile:
									valid_aupr = np.array(list(csv.reader(csvfile))).astype(np.float)
									valid_aupr = valid_aupr[:,1:]
							except:
								pass

							

							
							pred_files_dir = path.join(root,opt.dataset,model,'epochs')
							best_test_metrics = {}
							best_test_metrics['ACC'] = 0
							best_test_metrics['HA'] = 0
							best_test_metrics['ebF1'] = 0
							best_test_metrics['maF1'] = 0
							best_test_metrics['miF1'] = 0
							best_test_metrics['Avg'] = 0
							best_test_metrics['avgAUC'] = 0
							best_test_metrics['medAUC'] = 0
							best_test_metrics['avgAUPR'] = 0
							best_test_metrics['p_at_1'] = 0
							best_test_metrics['p_at_3'] = 0
							best_test_metrics['p_at_5'] = 0

							best_thresholds = {}
							best_thresholds['ACC'] = 0
							best_thresholds['HA'] = 0
							best_thresholds['ebF1'] = 0
							best_thresholds['maF1'] = 0
							best_thresholds['miF1'] = 0
							best_thresholds['Avg'] = 0

							best_valid_metrics = {}
							best_valid_metrics['ACC'] = 0
							best_valid_metrics['HA'] = 0
							best_valid_metrics['ebF1'] = 0
							best_valid_metrics['maF1'] = 0
							best_valid_metrics['miF1'] = 0
							best_valid_metrics['Avg'] = 0
							best_valid_metrics['avgAUC'] = 0
							best_valid_metrics['medAUC'] = 0
							best_valid_metrics['avgAUPR'] = 0
							best_valid_metrics['p_at_1'] = 0
							best_valid_metrics['p_at_3'] = 0
							best_valid_metrics['p_at_5'] = 0
							
							Flag = False
							for i in range(1,opt.max+1):

								try:
									if np.mean(valid_auc[i]) > best_valid_metrics['avgAUC']:
										best_valid_metrics['avgAUC'] = np.mean(valid_auc[i])
										best_test_metrics['avgAUC'] = np.mean(test_auc[i])
									if np.median(valid_auc[i]) > best_valid_metrics['medAUC']:
										best_valid_metrics['medAUC'] = np.median(valid_auc[i])
										best_test_metrics['medAUC'] = np.median(test_auc[i])
									if np.mean(valid_aupr[i]) > best_valid_metrics['avgAUPR']:
										best_valid_metrics['avgAUPR'] = np.mean(valid_aupr[i])
										best_test_metrics['avgAUPR'] = np.mean(test_aupr[i])
								except:
									pass


    							

							
								try:
									valid_predictions = load(path.join(pred_files_dir,'valid_preds'+str(i)+'.pt'))
									valid_targets = load(path.join(pred_files_dir,'valid_targets'+str(i)+'.pt'))
									
									test_predictions = load(path.join(pred_files_dir,'test_preds'+str(i)+'.pt'))
									test_targets = load(path.join(pred_files_dir,'test_targets'+str(i)+'.pt'))


									valid_pat1,valid_pat3,valid_pat5 = precision_at_k(valid_predictions,valid_targets)
									test_pat1,test_pat3,test_pat5 = precision_at_k(test_predictions,test_targets)
									if valid_pat1 > best_valid_metrics['p_at_1']:
										best_valid_metrics['p_at_1'] = valid_pat1
										best_test_metrics['p_at_1'] = test_pat1
									if valid_pat3 > best_valid_metrics['p_at_3']:
										best_valid_metrics['p_at_3'] = valid_pat3
										best_test_metrics['p_at_3'] = test_pat3
									if valid_pat5 > best_valid_metrics['p_at_5']:
										best_valid_metrics['p_at_5'] = valid_pat5
										best_test_metrics['p_at_5'] = test_pat5


									Flag = True
									
									for threshold in THRESHOLDS:
										opt.br_threshold = threshold

										valid_metrics = compute_metrics(valid_predictions.clone(),valid_targets.clone(),0,opt,0,all_metrics=False,verbose=False)
										test_metrics = compute_metrics(test_predictions.clone(),test_targets.clone(),0,opt,0,all_metrics=False,verbose=False)
										
										for metric in METRICS:
											if valid_metrics[metric] > best_valid_metrics[metric]:
												best_valid_metrics[metric] = valid_metrics[metric]
												best_test_metrics[metric] = test_metrics[metric]
												
												best_thresholds[metric] = threshold
																	
								except FileNotFoundError:
									break

							if (i > 49) or (('rcv1' in opt.dataset) and (i > 3)) or (('gm12878' in opt.dataset) and (i > 50)) or (('tcell' in opt.dataset) and (i > 49)):
								save(best_test_metrics,path.join(root,opt.dataset,model+'/final_results.pt'))
								print('file saved')

						if Flag is True:
							# for metric in best_thresholds:
							# 	print(best_thresholds[metric])
							# print(path.join(opt.dataset,model))
							if not loaded:
								print(path.join(opt.dataset,model))
								print('Last File Found at Epoch '+str(i-1))
							
							try:
								print('avgAUC: '+ str(best_test_metrics['avgAUC']))
							except:
								print('avgAUC: '+ str(best_test_metrics['meanAUC']))

							try:
								print('medAUC: '+ str(best_test_metrics['medAUC']))
								print('AUPR: '+ str(best_test_metrics['avgAUPR']))
								print('P@1: '+ str(best_test_metrics['p_at_1']))
								print('P@3: '+ str(best_test_metrics['p_at_3']))
								print('P@5: '+ str(best_test_metrics['p_at_5']))
							except:
								pass
							
							
							for metric in METRICS:
								print(metric+': '+str(best_test_metrics[metric]))
								csv_file.write(','+str("%.4f" % best_test_metrics[metric]))
								
							test_avg = np.array([best_test_metrics[metric] for metric in METRICS]).sum()/float(len(METRICS))
							print('Avg:  '+ str(test_avg))
							csv_file.write(','+str("%.4f" % test_avg))

							try:
								csv_file.write(','+str("%.4f" % best_test_metrics['avgAUC']))
							except:
								csv_file.write(','+str("%.4f" % best_test_metrics['meanAUC']))

							try: 
								csv_file.write(','+str("%.4f" % best_test_metrics['avgAUPR']))
								csv_file.write(','+str("%.4f" % best_test_metrics['p_at_1']))
								csv_file.write(','+str("%.4f" % best_test_metrics['p_at_3']))
								csv_file.write(','+str("%.4f" % best_test_metrics['p_at_5']))
							except: pass
							
							csv_file.write(','+str(i-1))
						csv_file.write('\n')
						print('')

	

if opt.max == -1:
	get_count()
else:
	from torch import load, save
	import numpy as np
	from evals import compute_metrics
	get_results()
