


import csv
import torch
import glob, os




factor_dict = {}

with open('hm_ids.csv') as csvfile:
	csv_reader = csv.DictReader(csvfile,delimiter=',',fieldnames=['internal ID','GEO ID','factor','cell line','tissue type'])
	csv_reader.next()
	for line in csv_reader:
		ID = line['internal ID']
		factor = line['factor']

		factor_dict[ID] = factor


dir ='HM/'
for pathAndFilename in glob.glob(dir+'/*.*'):
	ID, _ = os.path.splitext(os.path.basename(pathAndFilename))
	ID = ID.split('.')[0].replace('_treat','')
	ext = pathAndFilename.split(ID)[1].split('_treat')[1]
	factor = factor_dict[ID]
	new_pathAndFilename = os.path.join(dir, factor+'_'+ID+ext)
	print pathAndFilename
	print new_pathAndFilename
	os.rename(pathAndFilename, new_pathAndFilename)
