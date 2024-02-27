# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:37:52 2023

@author: xzhou
"""
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_recall_curve,auc, roc_auc_score
import numpy as np


def get_train_index(Y, train_ratio, val_ratio, test_ratio,n_splits,split_index):
    rs = ShuffleSplit(n_splits,train_size=train_ratio,test_size=1-train_ratio, random_state=0)
    train_indextrp=[]
    for train_index, test_index in rs.split(Y):
        train_indextrp.append(train_index)
       
    idx_train=train_indextrp[split_index]
    
    train_num, val_num, test_num = int(Y.shape[0]*train_ratio), int(Y.shape[0]*val_ratio), int(Y.shape[0]*test_ratio)
  
    idx_train, idx_val, idx_test, y_train, y_val, y_test, train_mask, val_mask, test_mask = get_splits(Y,idx_train, train_num, val_num, test_num, flag=False)

       
    return idx_train, idx_val, idx_test, y_train, y_val, y_test, train_mask, val_mask, test_mask

def get_splits(y, idx_train,tr_ratio, val_ratio, ts_ratio, flag=True):
    """
    flag:
    - True : tr_ratio, val_ratio, ts_ratio are ratios
    - False: tr_ratio, val_ratio, ts_ratio are ratios
    """

    N = y.shape[0]
    if flag:
        Ntr = int(N*tr_ratio)
        Nval = int(N*val_ratio)
    else:
        Ntr = tr_ratio
        Nval = val_ratio
 
    idx = list(set(range(y.shape[0]))-set(idx_train))

    if Nval>0:
        idx_val = list(np.random.choice(idx, Nval, replace=False))
        idx_test = list(set(idx)-set(idx_val))
    else:
        idx_val = None
        idx_test = idx
 
    if Ntr==0:
        y_train = None
        train_mask = None
    else:
        y_train = np.zeros(y.shape, dtype=np.int32)
        y_train[idx_train] = y[idx_train]
        train_mask = sample_mask(idx_train, N)

    if Nval==0:
        y_val = None
        val_mask = None
    else:
        y_val = np.zeros(y.shape, dtype=np.int32)
        y_val[idx_val] = y[idx_val]
        val_mask = sample_mask(idx_val, N)

    y_test = np.zeros(y.shape, dtype=np.int32)
    y_test[idx_test] = y[idx_test]
    test_mask = sample_mask(idx_test, N)
    
    return  idx_train, idx_val, idx_test, y_train, y_val, y_test, train_mask, val_mask, test_mask
 
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.float)

def get_AUC_PR(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true[:,1], y_prob[:,1]) # calculate precision-recall curve
    AUC_PR = auc(recall, precision)
    return AUC_PR 

def readListfile(path):
    IDList=[]
    with open(path) as f:
        for line in f:
            ID=line.replace('\n', '')            
            IDList.append(ID)
    
    return IDList 

def calculateauc(logits, y):
    y_prob = logits.cpu().detach().numpy()

    y=y.cpu().detach().numpy()
    AUC_ROC=roc_auc_score(y, y_prob)

    precision, recall, thresholds = precision_recall_curve(y, y_prob) # calculate precision-recall curve
    AUC_PR = auc(recall, precision)    
    
    return AUC_ROC,AUC_PR