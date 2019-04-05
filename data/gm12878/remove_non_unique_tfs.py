
from pdb import set_trace as stop


sorted_tfs_file = open('sorted_TFs_GM12878.csv',"r+")

unique_tf_list = []
uniqe_tf_ids = []

for line in sorted_tfs_file:
	tf_name_withid = line.strip()
	tf_name = tf_name_withid.split('_')[0]
	if tf_name not in unique_tf_list:
		unique_tf_list.append(tf_name)
		uniqe_tf_ids.append(tf_name_withid.upper())

 
for split in ['train','valid','test']:

	seqs_in = open(split+'_inputs.txt','r')
	tfs_in = open(split+'_labels.txt','r')

	seqs_out = open('../gm12878_unique2/'+split+'_inputs.txt','w')
	tfs_out = open('../gm12878_unique2/'+split+'_labels.txt','w')

	next_seq_in = seqs_in.readline()
	next_tfs_in = tfs_in.readline()

	while (next_seq_in and next_tfs_in):
		TFs = next_tfs_in.strip().split(' ')
		new_TFs = []
		for tf in TFs:
			# if tf.upper() in uniqe_tf_ids:
			# 	new_TFs.append(tf.upper())
			if tf.upper().split('_')[0] not in new_TFs:
				new_TFs.append(tf.upper().split('_')[0])
		if len(new_TFs) > 0:
			seqs_out.write(next_seq_in)
			for tf in new_TFs:
				tfs_out.write(tf+' ')
			tfs_out.write('\n')


		next_seq_in = seqs_in.readline()
		next_tfs_in = tfs_in.readline()
