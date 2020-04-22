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
from .plotting import draw_weighted_distributions, draw_unweighted_distributions, draw_ROC, resampled_discriminator_and_roc

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
        filename=None,
        plot=False,
        do = 'sherpaVsMG5',
        randomize = False,
        test_split=0.2,
        validation_split=0.2,
        partition="train",
        n_processes=1,
    ):
        """
        Parameters
        ----------
        folder : str or None
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format). Default value:
            None.
        filename : str or None
            Filenames for the resulting samples. A prefix such as 'x' as well as the extension
            '.npy' will be added automatically. Default value:
            None.
        test_split : float or None, optional
            Fraction of events reserved for the evaluation sample (that will not be used for any training samples).
            Default value: 0.2.
        n_processes : None or int, optional
            If None or larger than 1, multiprocessing will be used to parallelize the sampling. In this case,
            n_workers sets the number of jobs running in parallel, and None will use the number of CPUs. Default value:
            1.
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

        variables = ['VpT','Njets','j1pT', 'j2pT', 'HT','ptmiss', 'l1pT','Veta','j1eta','j2eta']
        vlabels = ['V $\mathrm{p_{T}}$ [GeV]','Number of jets','Leading jet $\mathrm{p_{T}}$ [GeV]','Subleading jet $\mathrm{p_{T}}$ [GeV]', '$\mathrm{H_{T}}$ [GeV]','$\mathrm{p_{T}^{miss}}$ [GeV]', '    Leading lepton $\mathrm{p_{T}}$ [GeV]','V $\eta$','Leading jet $\eta$','Subleading jet $\eta$']
        binning = [range(0, 2400, 200), range(0, 15, 1), range(0, 2700, 200),range(0, 2700, 200),range(0, 5000, 250),range(0, 600, 100),range(0, 1500, 100), etaV, etaJ, etaJ]

        # load sample X0
        if do == "sherpaVsMG5":
            x0 = load(filename = '/eos/user/m/mvesterb/data/sherpa/one/Nominal.root', variables = variables)
            x1 = load(filename = '/eos/user/m/mvesterb/data/madgraph/one/Nominal.root', variables = variables)
            legend = ["Sherpa","Madgraph"]
        else: 
            x0  = load(filename = '/eos/user/m/mvesterb/data/MUR1_MUF1_PDF261000.root', variables = variables)
            x1  = load(filename = '/eos/user/m/mvesterb/data/MUR2_MUF1_PDF261000.root', variables = variables)
            wx0 = load(filename = '/eos/user/m/mvesterb/data/MUR1_MUF1_PDF261000.root', variables = ['truthWeight'])
            wx1 = load(filename = '/eos/user/m/mvesterb/data/MUR2_MUF1_PDF261000.root', variables = ['truthWeight'])
            legend = ["MUR1", "MUR2"]
            pTruth0    = (wx0.truthWeight)/np.sum(wx0.truthWeight.astype(np.float))
            pTruth1    = (wx1.truthWeight)/np.sum(wx1.truthWeight.astype(np.float))
            iTruth0    = np.random.choice(np.arange(len(x0)),size=int(np.sum(wx0.truthWeight.astype(np.float))),p=pTruth0)
            iTruth1    = np.random.choice(np.arange(len(x1)),size=int(np.sum(wx1.truthWeight.astype(np.float))),p=pTruth1)
            x0     = x0.iloc[iTruth0] #original
            x1     = x1.iloc[iTruth1] #target


        # randomize training and test data (or not)
        n_target = x0.values.shape[0]
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
        if filename is not None and folder is not None:
            np.save(folder + do + "/x0_test.npy", X0_test)
            np.save(folder + do + "/x0_" + filename + ".npy", X0)
            np.save(folder + do + "/x1_" + filename + ".npy", X1)
            np.save(folder + do + "/x_" + filename + ".npy", x)
            np.save(folder + do + "/y_" + filename + ".npy", y)

        if plot:
            draw_unweighted_distributions(X0, X1, np.ones(X0[:,0].size), variables, vlabels, binning, legend) 
            print("saving plots")
            
        return x, y                                                                                                                                                                                                                                      

    def load_result(
        self,
        x0,
        x1,
        weights = None,
        label = None,
        do = 'sherpaVsMG5',
    ):
        """
        Parameters
        ----------
        weights : ndarray
            r_hat weights:
            None.
        Returns
        -------
        """
        etaV = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        etaJ = [-2.8,-2.4,-2,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,2,2.4,2.8]

        variables = ['VpT','Njets','j1pT', 'j2pT', 'HT','ptmiss', 'l1pT','Veta','j1eta','j2eta']
        vlabels = ['V $\mathrm{p_{T}}$ [GeV]','Number of jets','Leading jet $\mathrm{p_{T}}$ [GeV]','Subleading jet $\mathrm{p_{T}}$ [GeV]', '$\mathrm{H_{T}}$ [GeV]','$\mathrm{p_{T}^{miss}}$ [GeV]', 'Leading lepton $\mathrm{p_{T}}$ [GeV]','V $\eta$','Leading jet $\eta$','Subleading jet $\eta$']
        binning = [range(0, 2400, 200), range(0, 15, 1), range(0, 2700, 200),range(0, 2700, 200),range(0, 5000, 250),range(0, 600, 100),range(0, 1500, 100), etaV, etaJ, etaJ]
        if do == "sherpaVsMG5":
            legend = ["Sherpa","Madgraph"]
        else:    
            legend = ["MUR1", "MUR2"]
        

        # load sample X0
        X0 = load_and_check(x0, memmap_files_larger_than_gb=1.0)
        # load sample X1
        X1 = load_and_check(x1, memmap_files_larger_than_gb=1.0)
        # plot reweighted distributions      
        weights = weights / weights.sum() * len(X1)
        draw_weighted_distributions(X0, X1, weights, variables, vlabels, binning, label, legend) 
        # plot ROC curves     
        draw_ROC(X0, X1, weights, label, legend)
