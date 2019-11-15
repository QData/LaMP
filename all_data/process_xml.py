import csv
from pdb import set_trace as stop
import numpy as np
import os

# dataset = 'AmazonCat13K'
dataset = 'Wiki10'

test_input_file = open(os.path.join(dataset,'test_inputs.txt'),'w')
test_label_file = open(os.path.join(dataset,'test_labels.txt'),'w')

with open(os.path.join(dataset,dataset.lower()+'_test.txt')) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            num_samples = row[0]
            num_features = row[1]
            num_labels = row[2]
            print(f'Samples, Features, Labels: {", ".join(row)}')
            line_count += 1
        else:
            labels=row[0].split(',')
            features=row[1:]
            

            # features_vec = np.zeros(int(num_features))
            # indices = [int(feature.split(':')[0]) for feature in features]
            # vals = [float(feature.split(':')[1]) for feature in features]
            # features_vec = np.insert(features_vec, indices, vals)
            # features_str = ' '.join(map(str, features_vec))
            # test_input_file.write(features_str)

            test_input_file.write(str(num_features)+' ')
            for feature in features:
                test_input_file.write(str(feature)+' ')
            
            for label in labels:
                test_label_file.write(str(label)+' ')

            test_input_file.write('\n')
            test_label_file.write('\n')

            line_count += 1
        
        print(str(line_count)+'/'+str(num_samples))

test_input_file.close()
test_label_file.close()



# exit()
####################################################################################
#####################################################################################

train_input_file = open(os.path.join(dataset,'train_inputs.txt'),'w')
train_label_file = open(os.path.join(dataset,'train_labels.txt'),'w')

with open(os.path.join(dataset,dataset.lower()+'_train.txt')) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            num_samples = row[0]
            num_features = row[1]
            num_labels = row[2]
            print(f'Samples, Features, Labels: {", ".join(row)}')
            line_count += 1
        else:
            labels=row[0].split(',')
            features=row[1:]
            


            # features_vec = [0]*int(num_features)
            # for feature in features:
            #     idx,val=feature.split(':')
            #     features_vec[int(idx)] = float(val)
            # for feature in features_vec:
            #     train_input_file.write(str(feature)+' ')
            train_input_file.write(str(num_features)+' ')
            for feature in features:
                train_input_file.write(str(feature)+' ')
            
            for label in labels:
                train_label_file.write(str(label)+' ')

            
            train_input_file.write('\n')
            train_label_file.write('\n')

            line_count += 1
        
        print(str(line_count)+'/'+str(num_samples))

train_input_file.close()
train_label_file.close()