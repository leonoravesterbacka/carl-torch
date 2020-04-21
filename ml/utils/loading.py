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

        create_missing_folders([folder])

        variables = ['VpT','Njets','j1pT', 'j2pT', 'HT','ptmiss', 'l1pT']
        vlabels = ['V $\mathrm{p_{T}}$ [GeV]','Number of jets','Leading jet $\mathrm{p_{T}}$ [GeV]','Subleading jet $\mathrm{p_{T}}$ [GeV]', '$\mathrm{H_{T}}$ [GeV]','$\mathrm{p_{T}^{miss}}$ [GeV]', 'Leading lepton $\mathrm{p_{T}}$ [GeV]']
        binning = [range(0, 1000, 50), range(0, 10, 1), range(0, 500, 25),range(0, 500, 25),range(0, 1000, 50),range(0, 400, 25),range(0, 500, 25)]

        # load sample X0
        x0 = load(filename = '/eos/user/m/mvesterb/data/sherpa/one/Nominal.root', variables = variables)
        
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
        x1 = load(filename = '/eos/user/m/mvesterb/data/madgraph/one/Nominal.root', variables = variables)
        X1 = x1.to_numpy()
       
        # combine
        x = np.vstack([X0, X1])
        y = np.zeros(x.shape[0])
        y[X0.shape[0] :] = 1.0

        # y shape
        y = y.reshape((-1, 1))

        # save data
        if filename is not None and folder is not None:
            np.save(folder + "/x0_test.npy", X0_test)
            np.save(folder + "/x0_" + filename + ".npy", X0)
            np.save(folder + "/x1_" + filename + ".npy", X1)
            np.save(folder + "/x_" + filename + ".npy", x)
            np.save(folder + "/y_" + filename + ".npy", y)

        if plot:
            draw_unweighted_distributions(X0, X1, np.ones(X0[:,0].size), variables, vlabels, binning) 
            print("saving plots")
            
        return x, y                                                                                                                                                                                                                                      

    def load_result(
        self,
        x0,
        x1,
        weights = None,
        label = None,
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

        variables = ['VpT','Njets','j1pT', 'j2pT', 'HT','ptmiss', 'l1pT']
        vlabels = ['V $\mathrm{p_{T}}$ [GeV]','Number of jets','Leading jet $\mathrm{p_{T}}$ [GeV]','Subleading jet $\mathrm{p_{T}}$ [GeV]', '$\mathrm{H_{T}}$ [GeV]','$\mathrm{p_{T}^{miss}}$ [GeV]', 'Leading lepton $\mathrm{p_{T}}$ [GeV]']
        binning = [range(0, 1000, 50), range(0, 10, 1), range(0, 500, 25),range(0, 500, 25),range(0, 1000, 50),range(0, 400, 25),range(0, 500, 25)]

        # load sample X0
        X0 = load_and_check(x0, memmap_files_larger_than_gb=1.0)
        # load sample X1
        X1 = load_and_check(x1, memmap_files_larger_than_gb=1.0)
        # plot reweighted distributions       
        draw_weighted_distributions(X0, X1, weights, variables, vlabels, binning, label) 
        # plot ROC curves     
        draw_ROC(X0, X1, weights, label)
