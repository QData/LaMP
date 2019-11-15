from nltk.corpus import wordnet as wn
import os

import argparse

parser = argparse.ArgumentParser()


parser.add_argument('-dataset', type=str, default='bookmarks')
opt = parser.parse_args()


dataset = opt.dataset
path = os.path.join(dataset,'labels.txt')

output_path = os.path.join(dataset,'tf_interactions.tsv')
out_file = open(output_path,'w+') 

out_file.write('#node1\tnode2\n')

labels = [line.rstrip('\n') for line in open(path)]


pos_threshold = 0.25
neg_threshold = 0.0

for label in labels:

	syns = wn.synsets(label)
	if len(syns)>0:
		name = syns[0].name()
		synset = wn.synset(name)
		
		# for hypernym in synset.hypernyms():
		# 	synset1 = hypernym.name()

		for label2 in labels:
			syns2 = wn.synsets(label2)
			if len(syns2)>0:
				name2 = syns2[0].name()
				synset2 = wn.synset(name2)
				
				# print(synset.hypernyms())
				wup = synset.wup_similarity(synset2)

				if (wup is None) or (wup < neg_threshold):
					#Negative Correlation
					# print(label)
					# print(label2)
					# print(wup)
					print('----------')
					# out_file.write(label+'\t'+label2+'\n')
					# pass

				elif (wup > pos_threshold):
					print(label)
					print(label2)
					print(wup)
					print('----------')
					out_file.write(label+'\t'+label2+'\n')
		# break






	

