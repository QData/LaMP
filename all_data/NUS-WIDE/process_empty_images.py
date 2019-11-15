import os.path
import numpy as np
from pdb import set_trace as stop

# data_root = '/bigtemp/jjl5sw/NUS-WIDE/image/'
data_root = '/localtmp/jjl5sw/NUS-WIDE/image/'

with open('AllTags1k.txt') as f:
	all_tag_lines_1k = f.read().splitlines()
all_tag_lines_1k = list(map(str.strip, all_tag_lines_1k))


with open('AllTags81.txt') as f:
	all_tag_lines = f.read().splitlines()

all_tag_lines = list(map(str.strip, all_tag_lines))


NewImagelist = open('Imagelist_processed.txt','w')
NewAllTags81 = open('AllTags81_processed.txt','w')
# NewAllTags1k = open('AllTags1k_processed.txt','w')

with open('Imagelist.txt','r') as f:
	for idx,line in enumerate(f):
		line = line.strip()
		file_name = os.path.join(data_root,line)
	
		if os.path.isfile(file_name):
			tag_array = np.array(all_tag_lines[idx].split(' ')).astype(np.float)
			if tag_array.sum() > 0:
				print(line)
				NewImagelist.write(line+'\n')
				NewAllTags81.write(all_tag_lines[idx]+'\n')
				# NewAllTags1k.write(all_tag_lines_1k[idx]+'\n')

NewImagelist.close()
NewAllTags81.close()
# NewAllTags1k.close()