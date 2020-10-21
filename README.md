<img src="LaMP.png" width="47%" height="47%">

**Graph Neural Networks for Multi-Label Classification**<br/>
Jack Lanchantin, Arshdeep Sekhon, Yanjun Qi<br/>
ECML-PKDD 2019<br/>
[[paper]](https://arxiv.org/abs/1904.08049)
[[slides]](https://www.cs.virginia.edu/~jjl5sw/documents/LaMP_slides.pdf)
[[poster]](https://www.cs.virginia.edu/~jjl5sw/documents/LaMP_poster.pdf)

This repository contains a PyTorch implementation of LaMP from [Neural Message Passing for Multi-Label Classification
](https://arxiv.org/abs/1904.08049) (Lanchantin, Sekhon, and Qi 2019)

## Overview
In this paper, we propose Label Message Passing (LaMP) Networks to model the joint
prediction of multiple labels by treating labels as nodes on a graph. 
We use message passing neural networks (a generalization of graph neural networks) to implicitly model
the dependencies between labels conditioned on an input. 

## Requirement
- python 3.4+
- pytorch 0.2.0
- tqdm
- numpy


## Usage

### Process The Data
Download the data from: [http://www.cs.virginia.edu/~jjl5sw/data/lamp_datasets.tar.gz](http://www.cs.virginia.edu/~jjl5sw/data/lamp_datasets.tar.gz) (745M)
```bash
wget http://www.cs.virginia.edu/~jjl5sw/data/lamp_datasets.tar.gz
```

Untar into the current directory
```bash
tar -xvf lamp_datasets.tar.gz -C ./
```

<!--
### 1) Preprocess the data for a specific dataset
```bash
python preprocess.py -train_src data/reuters/train_inputs.txt -train_tgt data/reuters/train_labels.txt -valid_src data/reuters/valid_inputs.txt -valid_tgt data/reuters/valid_labels.txt -test_src data/reuters/test_inputs.txt -test_tgt data/reuters/test_labels.txt -save_data data/reuters/train_valid_test.pt -max_seq_len 300
```
-->

Note: the data directory provided includes preprocessed data. To use your own data, see utils/preprocess.py which includes an example how to run in the comments at the top.

### Train and Test the model (training script contains the validation and testing code)
```bash
python main.py -dataset reuters -batch_size 32 -d_model 512 -d_inner_hid 512 -n_layers_enc 2 -n_layers_dec 2 -n_head 4 -epoch 50 -dropout 0.2 -dec_dropout 0.2 -lr 0.0002 -encoder 'graph' -decoder 'graph' -label_mask 'prior'
```

All datasets allow for the usage of the graph encoder (`-encoder 'graph'`) except for nuswide_vector, which requires the encoder argument to be set to 'mlp' (`-encoder 'mlp'`), since this is a vector input (see Appendix Table 5 of the paper for all data types). 

To use the fully connected label graph, use `-label_mask none`, and to use the edgeless label graph, use `-label_mask inv_eye`.

The main.py file evaluates each epoch for all metrics using the default threshold. However, in order to get the final results, you need to find the optimal threshold for each metric on the validation set. We selected from the following thresholds:
[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.8,0.85,0.9,0.95]

Feel free to open an issue with any questions.

## Acknowledgement
Much of this code was adapted from https://github.com/jadore801120/attention-is-all-you-need-pytorch
