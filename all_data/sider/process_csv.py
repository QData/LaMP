import csv
from rdkit import Chem
import torch
from pdb import set_trace as stop



f = open('sider.csv', 'r')
reader = csv.reader(f)
headers = next(reader, None)


inputs_f = open('inputs.txt','w')
adj_matrices_f = open('adj_matrices.txt','w')
labels_f = open('labels.txt','w')

for row in reader:
	
	mol = Chem.MolFromSmiles(str(row[0]))
	if mol:
		for atom in mol.GetAtoms():
			inputs_f.write(str(atom.GetAtomicNum())+' ') 
		inputs_f.write('\n')

		matrix = torch.from_numpy(Chem.GetAdjacencyMatrix(mol))
		for value in matrix.view(-1):
			adj_matrices_f.write(str(value.item())+' ')
		adj_matrices_f.write('\n')

		for label_idx, label_val in enumerate(row[1:]):
			if int(label_val) == 1:
				labels_f.write(str(label_idx+1)+' ')
		labels_f.write('\n')

			

inputs_f.close()
labels_f.close()
adj_matrices_f.close()


# for atom in mol.GetAtoms():
# 	print(atom.GetAtomicNum())

# for atom in mol.GetAtoms():
# 	print([x.GetAtomicNum() for x in atom.GetNeighbors()])