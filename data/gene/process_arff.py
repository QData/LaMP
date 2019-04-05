from pdb import set_trace as stop
from torch import save

with open('label_ontology_noroot.csv', 'r') as myfile:
# with open('label_ontology_noroot.csv', 'r') as myfile:
	data=myfile.read().replace('\n', '')
pairs = data.split(',')

# label_list = []
# for pair in pairs:
# 	split_pair = pair.split('/')
# 	first = split_pair[0]
# 	if first not in label_list:
# 		label_list.append(first)
# 	if len(split_pair)>1:
# 		second = split_pair[1]
# 		if second not in label_list:
# 			label_list.append(second)

labels105 = ['GO0000027','GO0000067','GO0000070','GO0000071','GO0000074','GO0000082','GO0000086','GO0000087','GO0000122','GO0000154','GO0000278','GO0000280','GO0000282','GO0000283','GO0000398','GO0000902','GO0000910','GO0001403','GO0006066','GO0006259','GO0006260','GO0006261','GO0006270','GO0006281','GO0006289','GO0006298','GO0006310','GO0006319','GO0006325','GO0006338','GO0006347','GO0006348','GO0006351','GO0006355','GO0006357','GO0006360','GO0006364','GO0006365','GO0006366','GO0006367','GO0006368','GO0006396','GO0006397','GO0006402','GO0006412','GO0006413','GO0006414','GO0006457','GO0006461','GO0006464','GO0006468','GO0006487','GO0006508','GO0006511','GO0006513','GO0006605','GO0006607','GO0006608','GO0006609','GO0006610','GO0006611','GO0006623','GO0006796','GO0006886','GO0006887','GO0006888','GO0006891','GO0006893','GO0006897','GO0006906','GO0006950','GO0006970','GO0006974','GO0006979','GO0006997','GO0006999','GO0007001','GO0007010','GO0007015','GO0007020','GO0007046','GO0007049','GO0007059','GO0007067','GO0007124','GO0007126','GO0007131','GO0008283','GO0015031','GO0015986','GO0016070','GO0016071','GO0016072','GO0016192','GO0016568','GO0016573','GO0019538','GO0030036','GO0030163','GO0030490','GO0042254','GO0045045','GO0045449','GO0045944','GO0048193']

labels105u = ['go0000027','go0000067','go0000070','go0000071','go0000074','go0000082','go0000086','go0000087','go0000122','go0000154','go0000278','go0000280','go0000282','go0000283','go0000398','go0000902','go0000910','go0001403','go0006066','go0006259','go0006260','go0006261','go0006270','go0006281','go0006289','go0006298','go0006310','go0006319','go0006325','go0006338','go0006347','go0006348','go0006351','go0006355','go0006357','go0006360','go0006364','go0006365','go0006366','go0006367','go0006368','go0006396','go0006397','go0006402','go0006412','go0006413','go0006414','go0006457','go0006461','go0006464','go0006468','go0006487','go0006508','go0006511','go0006513','go0006605','go0006607','go0006608','go0006609','go0006610','go0006611','go0006623','go0006796','go0006886','go0006887','go0006888','go0006891','go0006893','go0006897','go0006906','go0006950','go0006970','go0006974','go0006979','go0006997','go0006999','go0007001','go0007010','go0007015','go0007020','go0007046','go0007049','go0007059','go0007067','go0007124','go0007126','go0007131','go0008283','go0015031','go0015986','go0016070','go0016071','go0016072','go0016192','go0016568','go0016573','go0019538','go0030036','go0030163','go0030490','go0042254','go0045045','go0045449','go0045944','go0048193']


forward_tree = {}
for pair in pairs:
	split_pair = pair.split('/')
	first = split_pair[0].lower()
	if first not in forward_tree:
		forward_tree[first] = []
	if len(split_pair)>1:
		second = split_pair[1].lower()
		if second not in forward_tree[first]:
			forward_tree[first].append(second)


for i in range(len(forward_tree)*len(forward_tree)):
	for key,value in forward_tree.items():
		for child in value:
			if child in forward_tree:
				for grandchild in forward_tree[child]:
					if not grandchild in value:
						forward_tree[key].append(grandchild.lower())

save(forward_tree,'forward_tree.pt')

reverse_tree = {}
for pair in pairs:
	split_pair = pair.split('/')
	first = split_pair[0]
	if len(split_pair)>1:
		second = split_pair[1]
		if second not in reverse_tree:
			reverse_tree[second] = []
		if first not in reverse_tree[second]:
			reverse_tree[second].append(first)
	else:
		if first not in reverse_tree:
			reverse_tree[first] = []

for i in range(len(reverse_tree)*len(reverse_tree)):
	for key,value in reverse_tree.items():
		for parent in value:
			for grandparent in reverse_tree[parent]:
				if not grandparent in value:
					reverse_tree[key].append(grandparent)



splits = ['train','valid','test']

label_list2 = []
for split in splits:
	read_file_name = 'borat_'+split+'.csv'
	feature_file = open(split+'_inputs.txt','w')
	label_file = open(split+'_labels.txt','w')

	with open(read_file_name, 'r') as file:
		lines = file.readlines()
		
	for line in lines:
		line = line.rstrip()

		inputs = line.split(',')[0:-1]
		labels = line.split(',')[-1].split('@')

		for feature in inputs:
			feature_file.write(feature+' ')
		feature_file.write('\n')


		all_labels = []
		for label in labels:
			if label in reverse_tree:
				if label not in all_labels:
					all_labels.append(label)
				for child in reverse_tree[label]:
					if child not in all_labels:
						all_labels.append(child)
					
		for label in all_labels:
		# for label in labels:
			label_file.write(label+' ')
		label_file.write('\n')

	feature_file.close()
	label_file.close()
		



