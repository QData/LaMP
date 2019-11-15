import csv
import glob, os, sys




cell_types = ['GM12878','Tcell']


for cell_type in cell_types:
	bed_directory = cell_type+'/TF/'
	for file in glob.glob(bed_directory+'/*.fa1'):
		print(file)
		p_values = []
		with open(file, "r") as fasta_file:
			f = open(file.replace('.fa1','.fa'), 'w')
			for line in fasta_file:
				if '>' in line:
					val = float(line.split('>')[1])
					p_values.append(val)
			p_values.sort()
			
			class1 = int(len(p_values) * (float(1)/3) )
			class1_threshold = p_values[class1-1]
			class2 = int(len(p_values) * (float(2)/3) )
			class2_threshold = p_values[class2-1]
			class3 = int(len(p_values) * (float(3)/3) )
			class3_threshold = p_values[class3-1]

			fasta_file.seek(0)
			for line in fasta_file:
				if '>' in line:
					val = float(line.split('>')[1])
					if val == float(0):
						f.write('>0\n')
					elif val <= class1_threshold:
						f.write('>1\n')
					elif val <= class2_threshold:
						f.write('>2\n')
					else:
						f.write('>3\n')
				else:
					f.write(line)
			f.close()
			sys.exit()




