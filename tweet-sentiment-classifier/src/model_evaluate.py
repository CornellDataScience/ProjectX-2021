# Author: Alexander Wang
# Updated: 11/8/2021

################################################################################
'''
 This package assumes y_real and y_pred are binarized values. 
 (1 for positive, 0 or -1 for negative)
 The purpose is to quantify the accuracy, precision, recall, and F1 scores. 
'''
import numpy as np

def sanity_check():
    print("model_evaluate.py successfully imported!")

def confusion_matrix(y_real, y_pred):
    assert(len(y_real) == len(y_pred))
     
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for yr, yp in zip(y_real, y_pred):
        if yr == yp: 
            if yp == 1:
                TP += 1
            else:
                TN += 1
        elif yr != yp:
            if yp == 1:
                FP += 1
            else:
                FN += 1
    
    return TP, TN, FP, FN

def accuracy(y_real, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_real, y_pred)
    return (TP + TN) / float(TP + TN + FP + FN)

def precision(y_real, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_real, y_pred)
    return TP / float(TP + FP)

def recall(y_real, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_real, y_pred)
    return TP / float(TP + FN)

def f1(y_real, y_pred):
    pre = precision(y_real, y_pred)
    rec = recall(y_real, y_pred)
    
    return 2.0 * (pre * rec) / (pre + rec)