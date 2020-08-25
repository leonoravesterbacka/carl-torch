from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import logging
import numpy as np
import pandas as pd
import seaborn as sns

from pandas.plotting import scatter_matrix
import multiprocessing
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
        do = 'sherpaVsMG5',
        x0 = None,
        x1 = None,
        randomize = False,
        save = False,
        correlation = True,
        preprocessing = True,
        nentries = 0,
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
        create_missing_folders(['plots'])

        variables = ['Njets','j1pT', 'j1eta','ptmiss','VpT','Veta']
        vlabels = ['Number of jets','Leading jet $\mathrm{p_{T}}$ [GeV]','Leading jet $\eta$','$\mathrm{p_{T}^{miss}}$ [GeV]','V $\mathrm{p_{T}}$ [GeV]','V $\eta$']
        etaJ = [-2.8,-2.4,-2,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,2,2.4,2.8]
        etaX = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11]

        # load samples
        if do == "sherpaVsMG5":
            legend = ["Sherpa","MG5"]
            if x0 is None and x1 is None: # if x0 and x1 are not provided, load them here
                x0 = load(filename = '/eos/user/m/mvesterb/data/sherpa/one/Nominal.root',   variables = variables, tree = 'tree_')
                x1 = load(filename = '/eos/user/m/mvesterb/data/madgraph/one/Nominal.root', variables = variables, tree = 'tree_')
        elif do == "qsf":
            legend = ["qsfUp", "qsfDown"]
            jetVariables = ['Njets', 'MET', 'Jet_Pt', 'Jet_Eta', 'Jet_Mass', 'Jet_Phi']
            lepVariables = ['Lepton_Pt', 'Lepton_Eta', 'Lepton_Mass', 'Lepton_Phi']
            vlabels = ['Number of jets', '$\mathrm{p_{T}^{miss}}$ [GeV]', 'Leading jet $\mathrm{p_{T}}$ [GeV]','Leading jet $\eta$', 'Leading jet mass [GeV]','Leading jet $\Phi$', 'Subleading jet $\mathrm{p_{T}}$ [GeV]','Subleading jet $\eta$', 'Subleading jet mass [GeV]','Subleading jet $\Phi$', 'Leading lepton $\mathrm{p_{T}}$ [GeV]','Leading lepton $\eta$', 'Leading lepton mass [GeV]','Leading lepton $\Phi$', 'Subleading lepton $\mathrm{p_{T}}$ [GeV]','Subleading lepton $\eta$', 'Subleading lepton mass [GeV]','Subleading lepton $\Phi$']
            etaX = [-2.8,-2.4,-2,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,2,2.4,2.8]
            jetBinning = [range(0, 2000, 200), etaJ, range(0, 1000, 100), etaJ]
            if x0 is None and x1 is None: # if x0 and x1 are not provided, load them here
                x0 = load(filename = '/afs/cern.ch/work/m/mvesterb/public/pmg/aug11/qsfUp/tree.root',   jets = jetVariables, leps = lepVariables, n = int(nentries), tree = 'Tree')
                x1 = load(filename = '/afs/cern.ch/work/m/mvesterb/public/pmg/aug11/qsfDown/tree.root', jets = jetVariables, leps = lepVariables, n = int(nentries), tree = 'Tree')
        binning = [range(0, 15, 1), range(0, 1000, 100)]+jetBinning+jetBinning+jetBinning+jetBinning

        if preprocessing:
            factor = 3
            for column in x0.columns:
                upper_lim = x0[column].mean () + x0[column].std () * factor
                upper_lim = x1[column].mean () + x1[column].std () * factor
                lower_lim = x0[column].mean () - x0[column].std () * factor
                lower_lim = x1[column].mean () - x1[column].std () * factor
                x0 = x0[(x0[column] < upper_lim) & (x0[column] > lower_lim)]
                x1 = x1[(x1[column] < upper_lim) & (x1[column] > lower_lim)]
            x0 = x0.round(decimals=2)
            x1 = x1.round(decimals=2)


        if correlation:
            cor0 = x0.corr()
            sns.heatmap(cor0, annot=True, cmap=plt.cm.Reds)
            cor_target = abs(cor0[variables[0]])
            relevant_features = cor_target[cor_target>0.5]
            print("relevant_features ", relevant_features)
            plt.savefig('plots/scatterMatrix_'+do+'.png')
            plt.clf()

        X0 = x0.to_numpy()
        X1 = x1.to_numpy()
        # combine
        x = np.vstack([X0, X1])
        y = np.zeros(x.shape[0])

        y[X0.shape[0] :] = 1.0
        # y shape
        y = y.reshape((-1, 1))
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=42)
        X_train, X_val,  y_train, y_val =  train_test_split(X_train, y_train, test_size=0.50, random_state=42)
        # save data
        if folder is not None:
            np.save(folder + do + "/x0_train.npy", X0)
            np.save(folder + do + "/x1_train.npy", X1)
            np.save(folder + do + "/x_train.npy", x)
            np.save(folder + do + "/y_train.npy", y)
            np.save(folder + do + "/X_train.npy", X_train)
            np.save(folder + do + "/X_val.npy", X_val)
            np.save(folder + do + "/Y_train.npy", y_train)
            np.save(folder + do + "/Y_val.npy", y_val)

        if plot:
            draw_unweighted_distributions(X0, X1, np.ones(X0[:,0].size), x0.columns, vlabels, binning, legend, save) 
            print("saving plots")
            
        return x, y                                                                                                                                                                                                                                      

    def load_result(
        self,
        x0,
        x1,
        weights = None,
        label = None,
        do = 'qsf',
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

        etaX = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11]
        etaJ = [-2.8,-2.4,-2,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,2,2.4,2.8]
        if do == "sherpaVsMG5":
            legend = ["Sherpa","MG5"]
        elif do == "qsf":
            etaX = [-2.8,-2.4,-2,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,2,2.4,2.8]
            legend = ["qsfUp", "qsfDown"]
            jetVariables = ['Njets', 'MET', 'Jet_Pt', 'Jet_Eta', 'Jet_Mass', 'Jet_Phi']
            lepVariables = ['Lepton_Pt', 'Lepton_Eta', 'Lepton_Mass', 'Lepton_Phi']
            vlabels = ['Number of jets', '$\mathrm{p_{T}^{miss}}$ [GeV]', 'Leading jet $\mathrm{p_{T}}$ [GeV]','Leading jet $\eta$', 'Leading jet mass [GeV]','Leading jet $\Phi$', 'Subleading jet $\mathrm{p_{T}}$ [GeV]','Subleading jet $\eta$', 'Subleading jet mass [GeV]','Subleading jet $\Phi$', 'Leading lepton $\mathrm{p_{T}}$ [GeV]','Leading lepton $\eta$', 'Leading lepton mass [GeV]','Leading lepton $\Phi$', 'Subleading lepton $\mathrm{p_{T}}$ [GeV]','Subleading lepton $\eta$', 'Subleading lepton mass [GeV]','Subleading lepton $\Phi$']
            etaX = [-2.8,-2.4,-2,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,2,2.4,2.8]
            jetBinning = [range(0, 2000, 200), etaJ, range(0, 1000, 100), etaJ]

        binning = [range(0, 15, 1), range(0, 1000, 100)]+jetBinning+jetBinning+jetBinning+jetBinning
        x0df = load(filename = '/afs/cern.ch/work/m/mvesterb/public/pmg/aug11/qsfUp/tree.root',   jets = jetVariables, leps = lepVariables, n = 1, tree = 'Tree')
        # load samples
        X0 = load_and_check(x0, memmap_files_larger_than_gb=1.0)
        X1 = load_and_check(x1, memmap_files_larger_than_gb=1.0)
        weights = weights / weights.sum() * len(X1)
        # plot ROC curves     
        draw_ROC(X0, X1, weights, label, legend, save)
        # plot reweighted distributions      
        draw_weighted_distributions(X0, X1, weights, x0df.columns, vlabels, binning, label, legend, save) 

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
