
import time
import torch
import csv

factor_dict = {}
with open('/af11/jjl5sw/chongzhi_datasets/GM12878/tf_ids.csv') as csvfile:
	csv_reader = csv.DictReader(csvfile,delimiter=',',fieldnames=['internal ID','GEO ID','factor','cell line','tissue type'])
	csv_reader.next()
	for line in csv_reader:
		ID = line['internal ID']
		factor = line['factor']
		factor_dict[ID] = factor


tf_dict = torch.load('tf_dict.pt')
all_tfs = torch.load('all_tfs.pt')
train = torch.Tensor(all_tfs['train'])

nz = torch.nonzero(train)
nz_windows = [:,0]
nz_tfs = [:,0]

co_binding_matrix = torch.zeros(train.size(1),train.size(1))

time1 = time.time()

num_windows = train.size(0)
k=0
for window in train:
    k+=1
    for i in range(window.size(0)):
        if int(window[i]) == 1:
            for j in range(i,window.size(0)):
                if int(window[j]) == 1:
                    co_binding_matrix[i][j] += 1
    print (float(k)/num_windows)



# num_windows = train.size(0)
# k=0
# for w in nz_windows.size(0)
#     window = train[nz_windows[w]]
#     k+=1
#     for i in range(window.size(0)):
#         if int(window[i]) == 1:
#             for j in range(i,window.size(0)):
#                 if int(window[j]) == 1:
#                     co_binding_matrix[i][j] += 1
#     print (float(k)/num_windows)



torch.save(co_binding_matrix,'co_binding_matrix.pt')

out_file = open('co_binding_matrix.csv','w')
out_file.write(',')
for k in tf_dict['train'].iterkeys():
    ID = factor_dict[k] + '_' +k
    out_file.write(ID+',')


out_file.write('\n')

i = 0
for k in tf_dict['train'].iterkeys():
    ID = factor_dict[k] + '_' +k
    out_file.write(ID+',')
    for j in range(co_binding_matrix.size(1)):
        out_file.write(str(co_binding_matrix[i][j])+',')
    out_file.write('\n')
    i +=1

out_file.close()
