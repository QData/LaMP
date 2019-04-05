
# coding: utf-8

# In[2]:


import csv
import numpy as np
from pdb import set_trace as stop
import torch


data_dict = torch.load('data_dict.pt')
interactions_file_name = 'tf_interactions.tsv'




def get_gm12878_adj_matrix(data_dict,interactions_file_name):

    interactions_file = csv.DictReader(open(interactions_file_name),delimiter='\t')

    interactions_dict = {}
    for line in interactions_file:
        node1 = line['#node1'].lower()
        node2 = line['node2'].lower()

        if not node1 in interactions_dict:
            interactions_dict[node1] = []
        interactions_dict[node1].append(node2)

        if not node2 in interactions_dict:
            interactions_dict[node2] = []
        interactions_dict[node2].append(node1)
            
        
    adjacency_matrix = torch.zeros((len(data_dict),len(data_dict)))


    for tf_name1 in data_dict.keys():
        for tf_name2 in data_dict.keys():
            if tf_name1 != tf_name2:
                tf_root1 = tf_name1.split('_')[0]
                tf_root2 = tf_name2.split('_')[0]

                if tf_root1 in interactions_dict:
                    if tf_root2 in interactions_dict[tf_root1]:
                        adjacency_matrix[int(data_dict[tf_name1])-4,int(data_dict[tf_name2])-4] = 1
                        adjacency_matrix[int(data_dict[tf_name2])-4,int(data_dict[tf_name1])-4] = 1

                if tf_root2 in interactions_dict:
                    if tf_root1 in interactions_dict[tf_root2]:
                        adjacency_matrix[int(data_dict[tf_name1])-4,int(data_dict[tf_name2])-4] = 1
                        adjacency_matrix[int(data_dict[tf_name2])-4,int(data_dict[tf_name1])-4] = 1

    
    print(adjacency_matrix)





def get_tf_adj_matrix(interactions_file_name):
    
    interactions_file = csv.DictReader(open(interactions_file_name),delimiter='\t')

    tf_list = []
    tf_dict = {}
    count = 0
    for tf in open('label_names.txt'):
        tf_list.append(tf.strip())
        tf_dict[tf.strip()] = count
        count +=1

    adjacency_matrix = np.zeros((len(tf_list),len(tf_list)))
    for line in interactions_file:
        node1 = line['#node1']
        node2 = line['node2']
        node1_id = int(tf_dict[node1])
        node2_id = int(tf_dict[node2])
        
        combined_score = line['combined_score']
        adjacency_matrix[node1_id,node2_id] = float(combined_score)
        adjacency_matrix[node2_id,node1_id] = float(combined_score)

    f_out = open('tf_adjacency_matrix.csv','w')

    for name in tf_dict:
        f_out.write(','+name)
    f_out.write('\n')

    count = 0
    for row in adjacency_matrix:
        f_out.write(tf_list[count])
        count +=1
        for value in row:
            f_out.write(','+str(value))
        f_out.write('\n')

    f_out.close()

        
        

get_gm12878_adj_matrix()