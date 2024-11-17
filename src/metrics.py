import torch
import torchvision

import pandas as pd
import numpy as np


def jaccard(y_true, y_pred):
    num = y_true.size(0)
    eps = 1e-7
    
    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)
    intersection = (y_true_flat * y_pred_flat).sum(1)
    union = ((y_true_flat + y_pred_flat) > 0.0).float().sum(1)
    
    score = (intersection) / (union + eps)
    score = score.sum() / num
    return score
    

def dice(y_true, y_pred):
    num = y_true.size(0)
    eps = 1e-7
    
    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)
    intersection = (y_true_flat * y_pred_flat).sum(1)
    
    score =  (2 * intersection) / (y_true_flat.sum(1) + y_pred_flat.sum(1) + eps)
    score = score.sum() / num
    return score

def specificity(y_true, y_pred):    
    num = y_true.size(0)
    eps = 1e-7
    
    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)
    
    true_negatives = ((1 - y_true_flat) * (1 - y_pred_flat)).sum(1)
    false_positives = ((1 - y_true_flat) * y_pred_flat).sum(1)
    
    specificity_score = true_negatives / (true_negatives + false_positives + eps)
    specificity_score = specificity_score.sum() / num
    return specificity_score


def precision(y_true, y_pred):    
    num = y_true.size(0)
    eps = 1e-7
    
    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)
    
    true_positives = (y_true_flat * y_pred_flat).sum(1)
    false_positives = ((1 - y_true_flat) * y_pred_flat).sum(1)
    
    precision_score = true_positives / (true_positives + false_positives + eps)
    precision_score = precision_score.sum() / num
    return precision_score


def recall(y_true, y_pred):    
    num = y_true.size(0)
    eps = 1e-7
    
    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)
    
    true_positives = (y_true_flat * y_pred_flat).sum(1)
    false_negatives = (y_true_flat * (1 - y_pred_flat)).sum(1)
    
    recall_score = true_positives / (true_positives + false_negatives + eps)
    recall_score = recall_score.sum() / num
    return recall_score


def f1_score(y_true, y_pred):    
    precision_score = precision(y_true, y_pred)
    recall_score = recall(y_true, y_pred)
    
    f1 = 2 * (precision_score * recall_score) / (precision_score + recall_score + 1e-7)
    return f1

def accuracy(y_true, y_pred):
    num = y_true.size(0)
    eps = 1e-7
    
    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)
    
    true_positives = (y_true_flat * y_pred_flat).sum(1)
    true_negatives = ((1 - y_true_flat) * (1 - y_pred_flat)).sum(1)
    total_pixels = y_true_flat.size(1)
    
    accuracy_score = (true_positives + true_negatives) / (total_pixels + eps)
    accuracy_score = accuracy_score.sum() / num
    return accuracy_score
