#!/usr/bin/python
# -*- coding=utf-8 -*-
"""
=======================================================
Evaluate methods: AUC, ACC, MCC, Kappa, AIC111
=======================================================

"""

# Code source: Zhice Fang

from sklearn import metrics
import numpy as np
from scipy.stats import ks_2samp

def mixup(x, y, alpha):
    candidates_data, candidates_label = x, y
#        offset = (step * batch_size) % (candidates_data.shape[0] - batch_size)
#        train_features_batch = candidates_data[offset:(offset + batch_size)]
#        train_labels_batch = candidates_label[offset:(offset + batch_size)]
    train_features_batch=x
    train_labels_batch=y
    shape=np.shape(train_features_batch)
    if alpha == 0:
        return train_features_batch, train_labels_batch
    if alpha > 0:
        weight = np.random.beta(alpha, alpha, shape[0])
        x_weight = weight.reshape(shape[0], 1,1)
        y_weight = weight.reshape(shape[0], 1)
        index = np.random.permutation(shape[0])
        x1, x2 = train_features_batch, train_features_batch[index]
        x = x1 * x_weight + x2 * (1 - x_weight)
        y1, y2 = train_labels_batch, train_labels_batch[index]
        y = y1 * y_weight + y2 * (1 - y_weight)
        return x, y

def data_aug_mixup(train_x, train_y, alpha, number):
    train_x_aug = train_x
    train_y_aug = train_y
    for i in range(number):
        x, y = mixup(train_x, train_y, alpha)
        train_x_aug = np.concatenate((train_x_aug, x), axis=0)
        train_y_aug = np.concatenate((train_y_aug, y), axis=0)
    return train_x_aug, train_y_aug

def pre_class(y_probability):
    pred_class = []
    for i in y_probability:
        if i > 0.5:
            pred_class.append(1)
        else:
            pred_class.append(0)
    return pred_class

def get_auc(y_real, y_probability):
    return metrics.roc_auc_score(y_real, y_probability)

def get_acc(y_real, y_probability):
    pred_class = pre_class(y_probability)
    return metrics.accuracy_score(y_real, pred_class)


def get_precision(y_real, y_probability):
    pred_class = pre_class(y_probability)
    return metrics.precision_score(y_real,pred_class)

def get_recall(y_real, y_probability):
    pred_class = pre_class(y_probability)
    return metrics.recall_score(y_real, pred_class)

def get_f1(y_real, y_probability):
    pred_class = pre_class(y_probability)
    return metrics.f1_score(y_real, pred_class)

def get_mcc(y_real, y_probability):
    pred_class = pre_class(y_probability)
    return metrics.matthews_corrcoef(y_real, pred_class)

def AIC(y_real, y_probability, k, n):
    '''赤池信息准则
    :param y_real:
    :param y_probability:
    :param k: number of features
    :param n: number of sample
    :return:
    '''
    pred_class = pre_class(y_probability)
    resid = y_real - pred_class
    SSR = sum(resid ** 2)
    # AICValue = 2*k+n*np.log(float(SSR)/n)
    AICValue = k*np.log(n) + n*np.log(float(SSR)/n)
    return AICValue

def get_RMSE(y_real, y_probability):
    pred_class = pre_class(y_probability)
    mse = metrics.mean_squared_error(y_real, pred_class)
    return mse**0.5

def get_MAE(y_real, y_probability):
    pred_class = pre_class(y_probability)
    mae = metrics.mean_absolute_error(y_real, pred_class)
    return mae

def get_kappa(y_real, y_probability):
    pred_class = pre_class(y_probability)
    kappa = metrics.cohen_kappa_score(y_real, pred_class)
    return kappa

def ks_calc_auc(y_real, y_probability_first):
    fpr, tpr, thresholds = metrics.roc_curve(y_real, y_probability_first)
    ks = max(tpr-fpr)
    return ks

def get_ROC(data_input_y,y_probability,save_path):
    fpr, tpr, thresholds = metrics.roc_curve(data_input_y, y_probability)
    fpr, tpr = fpr.tolist(), tpr.tolist()
    # print(fpr,type(fpr))
    with open(save_path, 'w') as fp:
        for num in range(len(fpr)):
            fp.write(str(fpr[num]) + ',' + str(tpr[num]) + '\n')

def get_IOA(y_real, y_probability):
    '''
    calculate the Index of Agreement
    :param y_pred:
    :param y_real:
    :return:
    '''
    y_pred = pre_class(y_probability)
    y_real_average = np.average(y_real)
    y_pred_average = np.average(y_pred)
    top = 0.0
    down = 0.0
    for i in range(len(y_real)):
        top += (y_pred[i] - y_real[i]) ** 2
        down += (np.fabs(y_real[i] - y_real_average) + np.fabs(y_pred[i] - y_pred_average)) ** 2

    d = 1 - top / down
    return d

def get_IOA1(y_real, y_probability):
    '''
    calculate the Index of Agreement
    :param y_pred:
    :param y_real:
    :return:
    '''

    y_real_average = np.average(y_real)
    y_pred_average = np.average(y_probability)
    top = 0.0
    down = 0.0
    for i in range(len(y_real)):
        top += (y_probability[i] - y_real[i]) ** 2
        down += (np.fabs(y_real[i] - y_real_average) + np.fabs(y_probability[i] - y_pred_average)) ** 2

    d = 1 - top / down
    return d


def get_MAPE(y_real, y_probability):
    result = 0.0
    number = len(y_real)
    for i in range(number):
        result += np.abs((y_real[i] - y_probability[i])/1.0)
    result = result*100/number
    return result