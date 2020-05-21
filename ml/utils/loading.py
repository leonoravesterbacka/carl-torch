from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import logging
import numpy as np
import root_numpy
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from functools import partial

from .tools import create_missing_folders, load, load_and_check
from .plotting import draw_weighted_distributions, draw_unweighted_distributions, draw_ROC, resampled_discriminator_and_roc, plot_calibration_curve

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
        do = 'sherpaVsMG5',
        x0 = None,
        x1 = None,
        randomize = False,
        save = False,
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

        create_missing_folders([folder+do])
        etaV = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        etaJ = [-2.8,-2.4,-2,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,2,2.4,2.8]

        variables = ['VpT','Njets','j1pT', 'j2pT', 'HT','ptmiss', 'l1pT','Veta', 'j1eta', 'j2eta']
        vlabels = ['V $\mathrm{p_{T}}$ [GeV]','Number of jets','Leading jet $\mathrm{p_{T}}$ [GeV]','Subleading jet $\mathrm{p_{T}}$ [GeV]', '$\mathrm{H_{T}}$ [GeV]','$\mathrm{p_{T}^{miss}}$ [GeV]', '    Leading lepton $\mathrm{p_{T}}$ [GeV]','V $\eta$','Leading jet $\eta$','Subleading jet $\eta$']

        # load samples
        if do == "sherpaVsMG5":
            legend = ["Sherpa","MG5"]
            binning = [range(0, 200, 20), range(0, 10, 1), range(0, 200, 20),range(0, 200, 20),range(0, 500, 50),range(0, 100, 10),range(0, 200, 20), etaV, etaJ, etaJ]
            if x0 is None and x1 is None: # if x0 and x1 are not provided, load them here
                x0 = load(filename = '/eos/user/m/mvesterb/data/sherpa/one/Nominal.root', variables = variables)
                x1 = load(filename = '/eos/user/m/mvesterb/data/madgraph/one/Nominal.root', variables = variables)
        elif do == "mur": 
            legend = ["MUR1", "MUR2"]
            binning = [range(0, 2400, 200), range(0, 15, 1), range(0, 2700, 200),range(0, 2700, 200),range(0, 5000, 250),range(0, 600, 100),range(0, 1500, 100), etaV, etaJ, etaJ]
            if x0 is None and x1 is None: # if x0 and x1 are not provided, load them here
                x0  = load(filename = '/eos/user/m/mvesterb/data/MUR1_MUF1_PDF261000.root', variables = variables)
                x1  = load(filename = '/eos/user/m/mvesterb/data/MUR2_MUF1_PDF261000.root', variables = variables)
                #for mur1 vs MUR2, resampling accoring to the generator (truth) weight is required
                wx0 = load(filename = '/eos/user/m/mvesterb/data/MUR1_MUF1_PDF261000.root', variables = ['truthWeight'])
                wx1 = load(filename = '/eos/user/m/mvesterb/data/MUR2_MUF1_PDF261000.root', variables = ['truthWeight'])
                p0    = (wx0.truthWeight)/np.sum(wx0.truthWeight.astype(np.float))
                p1    = (wx1.truthWeight)/np.sum(wx1.truthWeight.astype(np.float))
                i0    = np.random.choice(np.arange(len(x0)),size=int(np.sum(wx0.truthWeight.astype(np.float))),p=p0)
                i1    = np.random.choice(np.arange(len(x1)),size=int(np.sum(wx1.truthWeight.astype(np.float))),p=p1)
                x0     = x0.iloc[i0] #original
                x1     = x1.iloc[i1] #target
        elif do == "qsf":
            legend = ["qsf up", "qsf down"]
            binning = [range(0, 2400, 200), range(0, 15, 1), range(0, 2700, 200),range(0, 2700, 200),range(0, 5000, 250),range(0, 600, 100),range(0, 1500, 100), etaV, etaJ, etaJ]
            if x0 is None and x1 is None: # if x0 and x1 are not provided, load them here
                x0 = load(filename = '/eos/user/m/mvesterb/data/qsfup/Nominal.root', variables = variables)
                x1 = load(filename = '/eos/user/m/mvesterb/data/qsfdown/Nominal.root', variables = variables)
        # randomize training and test data (or not)
        n_target = x1.values.shape[0]
        if randomize:
            randomized = x0.values[np.random.choice(range(x0.values.shape[0]),2*n_target,replace=True)]
            X0  = randomized[:n_target,:]
            X0_test = randomized[n_target:,:]                                                                 
        else:
            X0 = x0.values[:n_target,:]
            X0_test = x0.values[-n_target:,:]

        # load sample X1
        X1 = x1.to_numpy()
       
        # combine
        x = np.vstack([X0, X1])
        y = np.zeros(x.shape[0])
        y[X0.shape[0] :] = 1.0
        # y shape
        y = y.reshape((-1, 1))

        # save data
        if folder is not None:
            np.save(folder + do + "/x0_test.npy",  X0_test)
            np.save(folder + do + "/x0_train.npy", X0)
            np.save(folder + do + "/x1_train.npy", X1)
            np.save(folder + do + "/x_train.npy", x)
            np.save(folder + do + "/y_train.npy", y)

        if plot:
            draw_unweighted_distributions(X0, X1, np.ones(X0[:,0].size), variables, vlabels, binning, legend, save) 
            print("saving plots")
            
        return x, y                                                                                                                                                                                                                                      

    def load_result(
        self,
        x0,
        x1,
        weights = None,
        label = None,
        do = 'sherpaVsMG5',
        save = False,
    ):
        """
        Parameters
        ----------
        weights : ndarray
            r_hat weights:
        Returns
        -------
        """
        etaV = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        etaJ = [-2.8,-2.4,-2,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,2,2.4,2.8]

        variables = ['VpT','Njets','j1pT', 'j2pT', 'HT','ptmiss', 'l1pT','Veta', 'j1eta', 'j2eta']
        vlabels = ['V $\mathrm{p_{T}}$ [GeV]','Number of jets','Leading jet $\mathrm{p_{T}}$ [GeV]','Subleading jet $\mathrm{p_{T}}$ [GeV]', '$\mathrm{H_{T}}$ [GeV]','$\mathrm{p_{T}^{miss}}$ [GeV]', 'Leading lepton $\mathrm{p_{T}}$ [GeV]','V $\eta$','Leading jet $\eta$','Subleading jet $\eta$']
        if do == "sherpaVsMG5":
            legend = ["Sherpa","MG5"]
            binning = [range(0, 200, 20), range(0, 10, 1), range(0, 200, 20),range(0, 200, 20),range(0, 500, 50),range(0, 100, 10),range(0, 200, 20), etaV, etaJ, etaJ]
        else:    
            legend = ["MUR1", "MUR2"]
            binning = [range(0, 2400, 200), range(0, 15, 1), range(0, 2700, 200),range(0, 2700, 200),range(0, 5000, 250),range(0, 600, 100),range(0, 1500, 100), etaV, etaJ, etaJ]
        

        # load samples
        X0 = load_and_check(x0, memmap_files_larger_than_gb=1.0)
        X1 = load_and_check(x1, memmap_files_larger_than_gb=1.0)
        weights = weights / weights.sum() * len(X1)
        # plot ROC curves     
        draw_ROC(X0, X1, weights, label, legend, save)
        # plot reweighted distributions      
        draw_weighted_distributions(X0, X1, weights, variables, vlabels, binning, label, legend, save) 

    def load_calibration(
        self,
        y_true,
        p1_raw = None,
        p1_cal = None,
        label = None,
        do = 'sherpaVsMG5',
        save = False
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
        plot_calibration_curve(y_true, p1_raw, p1_cal, do, save)                                                                                                                                                                                                                                                                   
