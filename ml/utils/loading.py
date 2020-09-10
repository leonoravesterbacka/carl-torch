from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import logging
import tarfile
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial

from .tools import create_missing_folders, load, load_and_check
from .plotting import draw_weighted_distributions, draw_unweighted_distributions, draw_ROC, resampled_discriminator_and_roc, plot_calibration_curve
from sklearn.model_selection import train_test_split
logger = logging.getLogger(__name__)


class Loader():
    """
    Loading of data.
    """
    def __init__(self):
        super(Loader, self).__init__()

    def loading(
        self,
        folder=None,
        plot=False,
        var = 'QSFUP',
        do  = 'dilepton',
        x0 = None,
        x1 = None,
        randomize = False,
        save = False,
        correlation = True,
        preprocessing = True,
        nentries = 0,
        path = '',
    ):
        """
        Parameters
        ----------
        folder : str or None
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format). Default value:
            None.
        plot : bool, optional
            make validation plots
        do : str
            Decide what samples to use. Can either be Sherpa Vs Madgraph ('sherpaVsMG5'), Renormalization scale up vs down ('mur') or qsf scale up vs down ('qsf') 
            Default value: 'sherpaVsMG5'
        x0 : dataframe of none
            Either pass a dataframe as in notebook, or None to load sample according to do option. 
        x1 : dataframe of none
            Either pass a dataframe as in notebook, or None to load sample according to do option. 
        randomize : bool, optional
            Randomize training sample. Default value: 
            False
        save : bool, optional
            Save training ans test samples. Default value:
            False
        Returns
        -------
        x : ndarray
            Observables with shape `(n_samples, n_observables)`. The same information is saved as a file in the given
            folder.
        y : ndarray
            Class label with shape `(n_samples, n_parameters)`. `y=0` (`1`) for events sample from the numerator
            (denominator) hypothesis. The same information is saved as a file in the given folder.
        """

        create_missing_folders([folder+'/'+do+'/'+var])
        create_missing_folders(['plots'])
        # load samples
        etaJ = [-2.8,-2.4,-2,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,2,2.4,2.8]
        eventVars = ['Njets', 'MET']
        jetVars   = ['Jet_Pt', 'Jet_Mass']
        lepVars   = ['Lepton_Pt']
        jetBinning = [range(0, 1500, 50), range(0, 200, 10)]
        lepBinning = [range(0, 700, 20)]
        x0, vlabels = load(f = path+'/Sh_228_ttbar_'+do+'_EnhMaxHTavrgTopPT_nominal.root', 
                           events = eventVars, jets = jetVars, leps = lepVars, n = int(nentries), t = 'Tree', do = do)
        x1, vlabels = load(f = path+'/Sh_228_ttbar_'+do+'_EnhMaxHTavrgTopPT_'+var+'.root', 
                           events = eventVars, jets = jetVars, leps = lepVars, n = int(nentries), t = 'Tree', do = do)
        binning = [range(0, 12, 1), range(0, 900, 25)]+jetBinning+jetBinning+lepBinning+lepBinning
        if preprocessing:
            factor = 5
            x00 = len(x0)
            x10 = len(x1)
            for column in x0.columns:
                upper_lim = x0[column].mean () + x0[column].std () * factor
                upper_lim = x1[column].mean () + x1[column].std () * factor
                lower_lim = x0[column].mean () - x0[column].std () * factor
                lower_lim = x1[column].mean () - x1[column].std () * factor
                x0 = x0[(x0[column] < upper_lim) & (x0[column] > lower_lim)]
                x1 = x1[(x1[column] < upper_lim) & (x1[column] > lower_lim)]
            x0 = x0.round(decimals=2)
            x1 = x1.round(decimals=2)
            print("filtered x0 outliers: ", (x00-len(x0))/len(x0)*100, "% ")
            print("filtered x1 outliers: ", (x10-len(x1))/len(x1)*100, "% ")


        if correlation:
            cor0 = x0.corr()
            sns.heatmap(cor0, annot=True, cmap=plt.cm.Reds)
            cor_target = abs(cor0[x0.columns[0]])
            relevant_features = cor_target[cor_target>0.5]
            print("relevant_features ", relevant_features)
            if plot:
                plt.savefig('plots/scatterMatrix_'+do+'_'+var+'.png')
                plt.clf()

        X0 = x0.to_numpy()
        X1 = x1.to_numpy()
        # combine
        y0 = np.zeros(x0.shape[0])
        y1 = np.ones(x1.shape[0])

        X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, test_size=0.40, random_state=42)
        X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.40, random_state=42)
        X0_train, X0_val,  y0_train, y0_val =  train_test_split(X0_train, y0_train, test_size=0.50, random_state=42)
        X1_train, X1_val,  y1_train, y1_val =  train_test_split(X1_train, y1_train, test_size=0.50, random_state=42)
        X_train = np.vstack([X0_train, X1_train])
        y_train = np.concatenate((y0_train, y1_train), axis=None)
        X_val   = np.vstack([X0_val, X1_val])
        y_val = np.concatenate((y0_val, y1_val), axis=None)
        print("y_val, ", y_val)
        # save data
        if folder is not None and save:
            np.save(folder + do + '/' + var + "/X_train_" +str(nentries)+".npy", X_train)
            np.save(folder + do + '/' + var + "/y_train_" +str(nentries)+".npy", y_train)
            np.save(folder + do + '/' + var + "/X_val_"   +str(nentries)+".npy", X_val)
            np.save(folder + do + '/' + var + "/y_val_"   +str(nentries)+".npy", y_val)
            np.save(folder + do + '/' + var + "/X0_val_"  +str(nentries)+".npy", X0_val)
            np.save(folder + do + '/' + var + "/X1_val_"  +str(nentries)+".npy", X1_val)
            np.save(folder + do + '/' + var + "/X0_train_"+str(nentries)+".npy", X0_train)
            np.save(folder + do + '/' + var + "/X1_train_"+str(nentries)+".npy", X1_train)
            #Tar data files if training is done on GPU
            if torch.cuda.is_available():
                plot = False #don't plot on GPU...
                tar = tarfile.open("data_out.tar.gz", "w:gz")
                for name in [folder + do + '/' + var + "/X_train_" +str(nentries)+".npy", 
                             folder + do + '/' + var + "/y_train_" +str(nentries)+".npy",
                             folder + do + '/' + var + "/X_val_"   +str(nentries)+".npy",
                             folder + do + '/' + var + "/y_val_"   +str(nentries)+".npy",
                             folder + do + '/' + var + "/X0_val_"  +str(nentries)+".npy",
                             folder + do + '/' + var + "/X1_val_"  +str(nentries)+".npy",
                             folder + do + '/' + var + "/X0_train_"+str(nentries)+".npy",
                             folder + do + '/' + var + "/X1_train_"+str(nentries)+".npy"]:
                    tar.add(name)
                tar.close()

        if plot and int(nentries) > 10000: # no point in plotting distributions with too few events
            draw_unweighted_distributions(X0, X1, np.ones(X0[:,0].size), x0.columns, vlabels, binning, var, do, nentries, plot) 
            print("saving plots")
        return X_train, y_train, X0_train, X1_train

    def load_result(
        self,
        x0,
        x1,
        weights = None,
        label = None,
        do = 'dilepton',
        var = 'qsf',
        plot = False,
        n = 0,
        path = '',
    ):
        """
        Parameters
        ----------
        weights : ndarray
            r_hat weights:
        Returns
        -------
        """
        eventVars = ['Njets', 'MET']
        jetVars   = ['Jet_Pt', 'Jet_Mass']
        lepVars   = ['Lepton_Pt']
        etaJ = [-2.8,-2.4,-2,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,2,2.4,2.8]
        jetBinning = [range(0, 1500, 50), range(0, 200, 10)]
        lepBinning = [range(0, 700, 20)]

        binning = [range(0, 12, 1), range(0, 900, 25)]+jetBinning+jetBinning+lepBinning+lepBinning
        x0df, labels = load(f = path+'/Sh_228_ttbar_'+do+'_EnhMaxHTavrgTopPT_nominal.root', 
                            events = eventVars, jets = jetVars, leps = lepVars, n = 1, t = 'Tree')
        # load samples
        X0 = load_and_check(x0, memmap_files_larger_than_gb=1.0)
        X1 = load_and_check(x1, memmap_files_larger_than_gb=1.0)
        weights = weights / weights.sum() * len(X1)
        if int(n) > 10000: # no point in plotting distributions with too few events, they only look bad 
            # plot ROC curves     
            draw_ROC(X0, X1, weights, label, var, do, plot)
            # plot reweighted distributions     
            draw_weighted_distributions(X0, X1, weights, x0df.columns, labels, binning, label, var, do, n, plot) 

    def load_calibration(
        self,
        y_true,
        p1_raw = None,
        p1_cal = None,
        label = None,
        do = 'dilepton',
        var = 'qsf',
        plot = False
    ):
        """
        Parameters
        ----------
        y_true : ndarray
            true targets
        p1_raw : ndarray
            uncalibrated probabilities of the positive class
        p1_cal : ndarray
            calibrated probabilities of the positive class
        Returns
        -------
        """

        # load samples
        y_true  = load_and_check(y_true,  memmap_files_larger_than_gb=1.0)
        plot_calibration_curve(y_true, p1_raw, p1_cal, do, var, plot)                                                                                                                                                                                                                                                                   
