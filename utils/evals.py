import numpy
import scipy.sparse as sp
import logging
from six.moves import xrange
from collections import OrderedDict
import sys
import pdb
from sklearn import metrics
from threading import Lock
from threading import Thread
import torch
import math
from pdb import set_trace as stop
import os
import pandas as pd
# import pylab as pl
from sklearn.metrics import roc_curve, auc

FORMAT = '[%(asctime)s] %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
LOGGER = logging.getLogger(__name__)


def list2sparse(A, n_labels=None):
    if n_labels is None:
        n_labels_ = 0
        for a in A:
            if n_labels_ < numpy.max(a):
                n_labels_ = numpy.max(a)
        n_labels = n_labels_

    n_samples = len(A)
    mat = sp.dok_matrix((n_samples, n_labels))
    for idx in xrange(n_samples):
        for item in A[idx]:
            mat[idx, item] = 1

    return mat.tocsr()


def is_sparse(matrix):
    return sp.issparse(matrix)


def is_binary_matrix(matrix):
    return numpy.all(numpy.logical_xor(matrix != 1, matrix != 0))


def sparse2dense(sparse_matrix):
    """ convert a sparse matrix into a dense matrix of 0 or 1.

    """
    assert sp.issparse(sparse_matrix)

    return numpy.asarray(sparse_matrix.toarray())


def prepare_evaluation(targets, preds):
    if is_sparse(targets):
        targets = sparse2dense(targets)

    if is_sparse(preds):
        preds = sparse2dense(preds)

    assert numpy.array_equal(targets.shape, preds.shape)
    assert is_binary_matrix(targets)
    assert is_binary_matrix(preds)

    return (targets, preds)


def subset_accuracy(true_targets, predictions, per_sample=False, axis=0):
    # print(true_targets.shape)
    # print(predictions.shape)
    result = numpy.all(true_targets == predictions, axis=axis)

    if not per_sample:
        result = numpy.mean(result)

    return result


def hamming_loss(true_targets, predictions, per_sample=False, axis=0):

    result = numpy.mean(numpy.logical_xor(true_targets, predictions),
                        axis=axis)

    if not per_sample:
        result = numpy.mean(result)

    return result


def compute_tp_fp_fn(true_targets, predictions, axis=0):
    # axis: axis for instance
    tp = numpy.sum(true_targets * predictions, axis=axis).astype('float32')
    fp = numpy.sum(numpy.logical_not(true_targets) * predictions,
                   axis=axis).astype('float32')
    fn = numpy.sum(true_targets * numpy.logical_not(predictions),
                   axis=axis).astype('float32')

    return (tp, fp, fn)


def example_f1_score(true_targets, predictions, per_sample=False, axis=0):
    tp, fp, fn = compute_tp_fp_fn(true_targets, predictions, axis=axis)

    numerator = 2*tp
    denominator = (numpy.sum(true_targets,axis=axis).astype('float32') + numpy.sum(predictions,axis=axis).astype('float32'))

    zeros = numpy.where(denominator == 0)[0]

    denominator = numpy.delete(denominator,zeros)
    numerator = numpy.delete(numerator,zeros)

    example_f1 = numerator/denominator


    if per_sample:
        f1 = example_f1
    else:
        f1 = numpy.mean(example_f1)

    return f1



def f1_score_from_stats(tp, fp, fn, average='micro'):
    assert len(tp) == len(fp)
    assert len(fp) == len(fn)

    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    if average == 'micro':
        f1 = 2*numpy.sum(tp) / \
            float(2*numpy.sum(tp) + numpy.sum(fp) + numpy.sum(fn))

    elif average == 'macro':

        def safe_div(a, b):
            """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
            with numpy.errstate(divide='ignore', invalid='ignore'):
                c = numpy.true_divide(a, b)
            return c[numpy.isfinite(c)]

        f1 = numpy.mean(safe_div(2*tp, 2*tp + fp + fn))

    return f1


def f1_score(true_targets, predictions, average='micro', axis=0):
    """
        average: str
            'micro' or 'macro'
        axis: 0 or 1
            label axis
    """
    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    tp, fp, fn = compute_tp_fp_fn(true_targets, predictions, axis=axis)
    f1 = f1_score_from_stats(tp, fp, fn, average=average)

    return f1






def compute_aupr_thread(all_targets,all_predictions):
    
    aupr_array = []
    lock = Lock()

    def compute_aupr_(start,end,all_targets,all_predictions):
        for i in range(all_targets.shape[1]):
            try:
                precision, recall, thresholds = metrics.precision_recall_curve(all_targets[:,i], all_predictions[:,i], pos_label=1)
                auPR = metrics.auc(recall,precision,reorder=True)
                lock.acquire() 
                aupr_array.append(numpy.nan_to_num(auPR))
                lock.release()
            except Exception: 
                pass
                 
    t1 = Thread(target=compute_aupr_, args=(0,100,all_targets,all_predictions) )
    t2 = Thread(target=compute_aupr_, args=(100,200,all_targets,all_predictions) )
    t3 = Thread(target=compute_aupr_, args=(200,300,all_targets,all_predictions) )
    t4 = Thread(target=compute_aupr_, args=(300,400,all_targets,all_predictions) )
    t5 = Thread(target=compute_aupr_, args=(400,500,all_targets,all_predictions) )
    t6 = Thread(target=compute_aupr_, args=(500,600,all_targets,all_predictions) )
    t7 = Thread(target=compute_aupr_, args=(600,700,all_targets,all_predictions) )
    t8 = Thread(target=compute_aupr_, args=(700,800,all_targets,all_predictions) )
    t9 = Thread(target=compute_aupr_, args=(800,900,all_targets,all_predictions) )
    t10 = Thread(target=compute_aupr_, args=(900,919,all_targets,all_predictions) )
    t1.start();t2.start();t3.start();t4.start();t5.start();t6.start();t7.start();t8.start();t9.start();t10.start()
    t1.join();t2.join();t3.join();t4.join();t5.join();t6.join();t7.join();t8.join();t9.join();t10.join()
    

    aupr_array = numpy.array(aupr_array)

    mean_aupr = numpy.mean(aupr_array)
    median_aupr = numpy.median(aupr_array)
    return mean_aupr,median_aupr,aupr_array

def compute_fdr(all_targets,all_predictions, fdr_cutoff=0.5):
    fdr_array = []
    for i in range(all_targets.shape[1]):
        try:
            precision, recall, thresholds = metrics.precision_recall_curve(all_targets[:,i], all_predictions[:,i],pos_label=1)
            fdr = 1- precision
            cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
            fdr_at_cutoff = recall[cutoff_index]
            if not math.isnan(fdr_at_cutoff):
                fdr_array.append(numpy.nan_to_num(fdr_at_cutoff))
        except: 
            pass
    
    fdr_array = numpy.array(fdr_array)
    mean_fdr = numpy.mean(fdr_array)
    median_fdr = numpy.median(fdr_array)
    var_fdr = numpy.var(fdr_array)
    return mean_fdr,median_fdr,var_fdr,fdr_array


def compute_aupr(all_targets,all_predictions):
    aupr_array = []
    for i in range(all_targets.shape[1]):
        try:
            precision, recall, thresholds = metrics.precision_recall_curve(all_targets[:,i], all_predictions[:,i], pos_label=1)
            auPR = metrics.auc(recall,precision,reorder=True)
            if not math.isnan(auPR):
                aupr_array.append(numpy.nan_to_num(auPR))
        except: 
            pass
    
    aupr_array = numpy.array(aupr_array)
    mean_aupr = numpy.mean(aupr_array)
    median_aupr = numpy.median(aupr_array)
    var_aupr = numpy.var(aupr_array)
    return mean_aupr,median_aupr,var_aupr,aupr_array



def compute_auc_thread(all_targets,all_predictions):
    
    auc_array = []
    lock = Lock()

    def compute_auc_(start,end,all_targets,all_predictions):
        for i in range(start,end):
            try:  
                auROC = metrics.roc_auc_score(all_targets[:,i], all_predictions[:,i])
                lock.acquire() 
                if not math.isnan(auROC):
                    auc_array.append(auROC)
                lock.release()
            except ValueError:
                pass
                
    t1 = Thread(target=compute_auc_, args=(0,100,all_targets,all_predictions) )
    t2 = Thread(target=compute_auc_, args=(100,200,all_targets,all_predictions) )
    t3 = Thread(target=compute_auc_, args=(200,300,all_targets,all_predictions) )
    t4 = Thread(target=compute_auc_, args=(300,400,all_targets,all_predictions) )
    t5 = Thread(target=compute_auc_, args=(400,500,all_targets,all_predictions) )
    t6 = Thread(target=compute_auc_, args=(500,600,all_targets,all_predictions) )
    t7 = Thread(target=compute_auc_, args=(600,700,all_targets,all_predictions) )
    t8 = Thread(target=compute_auc_, args=(700,800,all_targets,all_predictions) )
    t9 = Thread(target=compute_auc_, args=(800,900,all_targets,all_predictions) )
    t10 = Thread(target=compute_auc_, args=(900,919,all_targets,all_predictions) )
    t1.start();t2.start();t3.start();t4.start();t5.start();t6.start();t7.start();t8.start();t9.start();t10.start()
    t1.join();t2.join();t3.join();t4.join();t5.join();t6.join();t7.join();t8.join();t9.join();t10.join()
    
    auc_array = numpy.array(auc_array)

    mean_auc = numpy.mean(auc_array)
    median_auc = numpy.median(auc_array)
    return mean_auc,median_auc,auc_array


def compute_auc(all_targets,all_predictions):
    auc_array = []
    lock = Lock()

    for i in range(all_targets.shape[1]):
        try:  
            auROC = metrics.roc_auc_score(all_targets[:,i], all_predictions[:,i])
            auc_array.append(auROC)
        except ValueError:
            pass
    
    auc_array = numpy.array(auc_array)
    mean_auc = numpy.mean(auc_array)
    median_auc = numpy.median(auc_array)
    var_auc = numpy.var(auc_array)
    return mean_auc,median_auc,var_auc,auc_array


def Find_Optimal_Cutoff(all_targets, all_predictions):
    thresh_array = []
    for j in range(all_targets.shape[1]):
        try:
            fpr, tpr, threshold = roc_curve(all_targets[:,j], all_predictions[:,j], pos_label=1)
            i = numpy.arange(len(tpr)) 
            roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
            roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]
            thresh_array.append(list(roc_t['threshold'])[0])
            
        except: 
            pass
    return thresh_array


def compute_metrics(all_predictions,all_targets,loss,args,elapsed,all_metrics=True,verbose=True):
    all_targets = all_targets.numpy()
    all_predictions = all_predictions.numpy()


    

    if all_metrics:
        meanAUC,medianAUC,varAUC,allAUC = compute_auc(all_targets,all_predictions)
        meanAUPR,medianAUPR,varAUPR,allAUPR = compute_aupr(all_targets,all_predictions)
        meanFDR,medianFDR,varFDR,allFDR = compute_fdr(all_targets,all_predictions)
    else:
        meanAUC,medianAUC,varAUC,allAUC = 0,0,0,0
        meanAUPR,medianAUPR,varAUPR,allAUPR = 0,0,0,0
        meanFDR,medianFDR,varFDR,allFDR = 0,0,0,0


    optimal_threshold = args.br_threshold
    
    # optimal_thresholds = Find_Optimal_Cutoff(all_targets,all_predictions)
    # optimal_threshold = numpy.mean(numpy.array(optimal_thresholds))
    

    if args.decoder in ['mlp','rnn_b','graph']:
        all_predictions[all_predictions < optimal_threshold] = 0
        all_predictions[all_predictions >= optimal_threshold] = 1
    else:
        all_predictions[all_predictions > 0.0] = 1 

    
        
    acc_ = list(subset_accuracy(all_targets, all_predictions, axis=1, per_sample=True))
    hl_ = list(hamming_loss(all_targets, all_predictions, axis=1, per_sample=True))
    exf1_ = list(example_f1_score(all_targets, all_predictions, axis=1, per_sample=True))        
    acc = numpy.mean(acc_)
    hl = numpy.mean(hl_)
    exf1 = numpy.mean(exf1_)
    

    tp, fp, fn = compute_tp_fp_fn(all_targets, all_predictions, axis=0)
    mif1 = f1_score_from_stats(tp, fp, fn, average='micro')
    maf1 = f1_score_from_stats(tp, fp, fn, average='macro')

    

    eval_ret = OrderedDict([('Subset accuracy', acc),
                        ('Hamming accuracy', 1 - hl),
                        ('Example-based F1', exf1),
                        ('Label-based Micro F1', mif1),
                        ('Label-based Macro F1', maf1)])

    
    ACC = eval_ret['Subset accuracy']
    HA = eval_ret['Hamming accuracy']
    ebF1 = eval_ret['Example-based F1']
    miF1 = eval_ret['Label-based Micro F1']
    maF1 = eval_ret['Label-based Macro F1']
    if verbose:
        print('ACC:   '+str(ACC))
        print('HA:    '+str(HA))
        print('ebF1:  '+str(ebF1))
        print('miF1:  '+str(miF1))
        print('maF1:  '+str(maF1))


    
    if verbose:
        print('uAUC:  '+str(meanAUC))
        # print('mAUC:  '+str(medianAUC))
        print('uAUPR: '+str(meanAUPR))
        # print('mAUPR: '+str(medianAUPR))
        print('uFDR: '+str(meanFDR))
        # print('mFDR:  '+str(medianFDR))

    metrics_dict = {}
    metrics_dict['ACC'] = ACC
    metrics_dict['HA'] = HA
    metrics_dict['ebF1'] = ebF1
    metrics_dict['miF1'] = miF1
    metrics_dict['maF1'] = maF1
    metrics_dict['meanAUC'] = meanAUC
    metrics_dict['medianAUC'] = medianAUC
    metrics_dict['meanAUPR'] = meanAUPR
    metrics_dict['allAUC'] = allAUC
    metrics_dict['medianAUPR'] = medianAUPR
    metrics_dict['allAUPR'] = allAUPR
    metrics_dict['meanFDR'] = meanFDR
    metrics_dict['medianFDR'] = medianFDR
    metrics_dict['loss'] = loss
    metrics_dict['time'] = elapsed

    return metrics_dict


class Logger:
    def __init__(self,args):
        self.model_name = args.model_name

        if args.model_name:
            try:
                os.makedirs(args.model_name)
            except OSError as exc:
                pass

            try:
                os.makedirs(args.model_name+'/epochs/')
            except OSError as exc:
                pass

            self.file_names = {}
            self.file_names['train'] = os.path.join(args.model_name,'train_results.csv')
            self.file_names['valid'] = os.path.join(args.model_name,'valid_results.csv')
            self.file_names['test'] = os.path.join(args.model_name,'test_results.csv')

            self.file_names['valid_all_aupr'] = os.path.join(args.model_name,'valid_all_aupr.csv')
            self.file_names['valid_all_auc'] = os.path.join(args.model_name,'valid_all_auc.csv')
            self.file_names['test_all_aupr'] = os.path.join(args.model_name,'test_all_aupr.csv')
            self.file_names['test_all_auc'] = os.path.join(args.model_name,'test_all_auc.csv')
            

            f = open(self.file_names['train'],'w+'); f.close()
            f = open(self.file_names['valid'],'w+'); f.close()
            f = open(self.file_names['test'],'w+'); f.close()
            f = open(self.file_names['valid_all_aupr'],'w+'); f.close()
            f = open(self.file_names['valid_all_auc'],'w+'); f.close()
            f = open(self.file_names['test_all_aupr'],'w+'); f.close()
            f = open(self.file_names['test_all_auc'],'w+'); f.close()
            os.utime(args.model_name,None)
        
        self.best_valid = {'loss':1000000,'ACC':0,'HA':0,'ebF1':0,'miF1':0,'maF1':0,'meanAUC':0,'medianAUC':0,'meanAUPR':0,'medianAUPR':0,'meanFDR':0,'medianFDR':0,'allAUC':None,'allAUPR':None}

        self.best_test = {'loss':1000000,'ACC':0,'HA':0,'ebF1':0,'miF1':0,'maF1':0,'meanAUC':0,'medianAUC':0,'meanAUPR':0,'medianAUPR':0,'meanFDR':0,'medianFDR':0,'allAUC':None,'allAUPR':None,'epoch':0}


    def evaluate(self,train_metrics,valid_metrics,test_metrics,epoch,num_params):
        if self.model_name:
            # if train_metrics is not None:
            #     with open(self.file_names['train'],'a') as f:
            #         f.write(str(epoch)+','+str(train_metrics['loss'])
            #                           +','+str(train_metrics['ACC'])
            #                           +','+str(train_metrics['HA'])
            #                           +','+str(train_metrics['ebF1'])
            #                           +','+str(train_metrics['miF1'])
            #                           +','+str(train_metrics['maF1'])
            #                           +','+str(train_metrics['meanAUC'])
            #                           +','+str(train_metrics['medianAUC'])
            #                           +','+str(train_metrics['meanAUPR'])
            #                           +','+str(train_metrics['medianAUPR'])
            #                           +','+str(train_metrics['meanFDR'])
            #                           +','+str(train_metrics['medianFDR'])
            #                           +','+'{elapse:3.3f}'.format(elapse=train_metrics['time'])
            #                           +','+str(num_params)
            #                           +'\n')
            
            # with open(self.file_names['valid'],'a') as f:
            #     f.write(str(epoch)+','+str(valid_metrics['loss'])
            #                       +','+str(valid_metrics['ACC'])
            #                       +','+str(valid_metrics['HA'])
            #                       +','+str(valid_metrics['ebF1'])
            #                       +','+str(valid_metrics['miF1'])
            #                       +','+str(valid_metrics['maF1'])
            #                       +','+str(valid_metrics['meanAUC'])
            #                       +','+str(valid_metrics['medianAUC'])
            #                       +','+str(valid_metrics['meanAUPR'])
            #                       +','+str(valid_metrics['medianAUPR'])
            #                       +','+str(valid_metrics['meanFDR'])
            #                       +','+str(valid_metrics['medianFDR'])
            #                       +','+'{elapse:3.3f}'.format(elapse=train_metrics['time'])
            #                       +','+'{elapse:3.3f}'.format(elapse=valid_metrics['time'])
            #                       +','+str(num_params)
            #                       +'\n')

            # with open(self.file_names['test'],'a') as f:
            #     f.write(str(epoch)+','+str(test_metrics['loss'])
            #                       +','+str(test_metrics['ACC'])
            #                       +','+str(test_metrics['HA'])
            #                       +','+str(test_metrics['ebF1'])
            #                       +','+str(test_metrics['miF1'])
            #                       +','+str(test_metrics['maF1'])
            #                       +','+str(test_metrics['meanAUC'])
            #                       +','+str(test_metrics['medianAUC'])
            #                       +','+str(test_metrics['meanAUPR'])
            #                       +','+str(test_metrics['medianAUPR'])
            #                       +','+str(test_metrics['meanFDR'])
            #                       +','+str(test_metrics['medianFDR'])
            #                       +','+'{elapse:3.3f}'.format(elapse=train_metrics['time'])
            #                       +','+'{elapse:3.3f}'.format(elapse=test_metrics['time'])
            #                       +','+str(num_params)
            #                       +'\n')


            with open(self.file_names['valid_all_auc'],'a') as f:
                f.write(str(epoch))
                for i,val in enumerate(valid_metrics['allAUC']):
                    f.write(','+str(val))
                f.write('\n')
                f.close()

            with open(self.file_names['valid_all_aupr'],'a') as f:
                f.write(str(epoch))
                for i,val in enumerate(valid_metrics['allAUPR']):
                    f.write(','+str(val))
                f.write('\n')
                f.close()

            with open(self.file_names['test_all_auc'],'a') as f:
                f.write(str(epoch))
                for i,val in enumerate(test_metrics['allAUC']):
                    f.write(','+str(val))
                f.write('\n')
                f.close()

            with open(self.file_names['test_all_aupr'],'a') as f:
                f.write(str(epoch))
                for i,val in enumerate(test_metrics['allAUPR']):
                    f.write(','+str(val))
                f.write('\n')
                f.close()


        for metric in valid_metrics.keys():
            if not 'all' in metric and not 'time'in metric:
                if  valid_metrics[metric] >= self.best_valid[metric]:
                    self.best_valid[metric]= valid_metrics[metric]
                    self.best_test[metric]= test_metrics[metric]
                    if metric == 'ACC':
                        self.best_test['epoch'] = epoch

         
        print('\n')
        print('**********************************')
        print('best ACC:  '+str(self.best_test['ACC']))
        print('best HA:   '+str(self.best_test['HA']))
        print('best ebF1: '+str(self.best_test['ebF1']))
        print('best miF1: '+str(self.best_test['miF1']))
        print('best maF1: '+str(self.best_test['maF1']))
        print('best meanAUC:  '+str(self.best_test['meanAUC']))
        print('best meanAUPR: '+str(self.best_test['meanAUPR']))
        print('best meanFDR: '+str(self.best_test['meanFDR']))
        print('**********************************')

        return self.best_valid,self.best_test
