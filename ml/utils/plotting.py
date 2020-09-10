from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPRegressor
from sklearn.calibration import calibration_curve


import torch
from .tools import create_missing_folders

logger = logging.getLogger(__name__)
hist_settings0 = {'alpha': 0.3}
hist_settings1 = {'histtype':'step', 'color':'black', 'linewidth':1, 'linestyle':'--'}

def draw_unweighted_distributions(x0, x1, weights, variables, vlabels, binning, legend, do, n, save = False):
    plt.figure(figsize=(14, 10))
    columns = range(len(variables))
    for id, column in enumerate(columns, 1):
        if save: plt.figure(figsize=(5, 4.2)) 
        else: plt.subplot(3,4, id)
        plt.yscale('log')
        plt.hist(x0[:,column], bins = binning[id-1], weights=weights, label = "nominal", **hist_settings0)
        plt.hist(x1[:,column], bins = binning[id-1], label = legend, **hist_settings1)
        plt.xlabel('ttbar %s, %s'%(do, vlabels[id-1]), horizontalalignment='right',x=1) 
        plt.legend(frameon=False)
        axes = plt.gca()
        axes.set_ylim([len(x0)*0.001,len(x0)*2])                  
        if save:
            create_missing_folders(["plots"])                                                              
            plt.savefig("plots/%s_%s_nominalVs%s_%s.png"%(variables[id-1], do, legend, n))                                                                
            plt.clf()
            plt.close()

def draw_weighted_distributions(x0, x1, weights, variables, vlabels, binning, label, legend, do, n, save = False):
    plt.figure(figsize=(14, 10))
    columns = range(len(variables))
    for id, column in enumerate(columns, 1):
        if save: plt.figure(figsize=(5, 4)) 
        else: plt.subplot(3,4, id)
        plt.yscale('log')
        plt.hist(x0[:,column], bins = binning[id-1], label = "nominal", **hist_settings0)
        plt.hist(x0[:,column], bins = binning[id-1], weights=weights, label = 'nominal*CARL', **hist_settings0)
        plt.hist(x1[:,column], bins = binning[id-1], label = legend, **hist_settings1)
        plt.xlabel('ttbar %s, %s'%(do, vlabels[id-1]), horizontalalignment='right',x=1) 
        plt.legend(frameon=False,title = '%s sample'%(label) )
        axes = plt.gca()
        axes.set_ylim([len(x0)*0.001,len(x0)*2])                 
        if save:
            create_missing_folders(["plots"])                                                              
            plt.savefig("plots/w_%s_%s_nominalVs%s_%s_%s.png"%(variables[id-1], do, legend,label, n)) 
            plt.clf()
            plt.close()

def weight_data(x0,x1,weights, max_weight=10000.):
    x1_len = x1.shape[0]
    x0_len = x0.shape[0]
    weights[weights>max_weight]=max_weight
    weights = weights / weights.sum()
    weighted_data = np.random.choice(range(x0_len), x0_len, p = weights)
    w_x0 = x0.copy()[weighted_data]
    y = np.zeros(x1_len + x0_len)
    x_all = np.vstack((w_x0,x1))
    y_all = np.zeros(x1_len +x0_len)
    y_all[x0_len:] = 1
    return (x_all,y_all)

def resampled_discriminator_and_roc(original, target, weights):
    (data, labels) = weight_data(original,target,weights)
    W = np.concatenate([weights / weights.sum() * len(target), [1] * len(target)])

    Xtr, Xts, Ytr, Yts, Wtr, Wts = train_test_split(data, labels, W, random_state=42, train_size=0.51, test_size=0.49)    
    
    discriminator = MLPRegressor(tol=1e-05, activation="logistic", 
               hidden_layer_sizes=(10, 10), learning_rate_init=1e-07, 
               learning_rate="constant", solver="lbfgs", random_state=1, 
               max_iter=75)

    discriminator.fit(Xtr,Ytr)
    predicted = discriminator.predict(Xts)
    fpr, tpr, _  = roc_curve(Yts,predicted.ravel())
    roc_auc = auc(fpr, tpr)
    return fpr,tpr,roc_auc
 
def draw_ROC(X0, X1, weights, label, legend, do, n, plot = True):
    plt.figure(figsize=(4, 3))
    no_weights_scaled = np.ones(X0.shape[0])/np.ones(X0.shape[0]).sum() * len(X1)
    fpr_t,tpr_t,roc_auc_t = resampled_discriminator_and_roc(X0, X1, no_weights_scaled)
    plt.plot(fpr_t, tpr_t, label=r"no weight, AUC=%.3f" % roc_auc_t)
    fpr_tC,tpr_tC,roc_auc_tC = resampled_discriminator_and_roc(X0, X1, weights)
    plt.plot(fpr_tC, tpr_tC, label=r"CARL weight, AUC=%.3f" % roc_auc_tC)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0]) 
    plt.ylim([0.0, 1.05])
    plt.title('Resampled proportional to weights')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", title = label)
    plt.tight_layout()
    if plot:
        plt.savefig('plots/roc_nominalVs%s_%s_%s_%s.png'%(legend,do,label, n)) 
        plt.clf()    
    logger.info("CARL weighted %s AUC is %.3f"%(label,roc_auc_tC))
    logger.info("Unweighted %s AUC is %.3f"%(label,roc_auc_t))
    logger.info("Saving ROC plots to /plots")

def plot_calibration_curve(y, probs_raw, probs_cal, do, var, save = False):
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    frac_of_pos_raw, mean_pred_value_raw = calibration_curve(y, probs_raw, n_bins=50)
    frac_of_pos_cal, mean_pred_value_cal = calibration_curve(y, probs_cal, n_bins=50)

    ax1.plot(mean_pred_value_raw, frac_of_pos_raw, "s-", label='uncalibrated', **hist_settings0)
    ax1.plot(mean_pred_value_cal, frac_of_pos_cal, "s-", label='calibrated', **hist_settings0)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'Calibration plot')
    
    ax2.hist(probs_raw, range=(0, 1), bins=50, label='uncalibrated', lw=2, **hist_settings0)
    ax2.hist(probs_cal, range=(0, 1), bins=50, label='calibrated', lw=2, **hist_settings0)
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count") 
    if save:
        plt.savefig('plots/calibration_'+do+'_'+var+'.png')
        plt.clf() 
    logger.info("Saving calibration curves to /plots")
