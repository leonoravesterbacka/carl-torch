from __future__ import absolute_import, division, print_function, unicode_literals

import six
import logging
import os
import stat
import numpy as np
import uproot
import uproot4
import pandas as pd
import torch
from torch.nn import functional as F

from contextlib import contextmanager

logger = logging.getLogger(__name__)

initialized = False

def load(f = None, events = None, jets = None, leps = None, n = 0, t = None, do = "dilepton"):
    if f is None:
        return None
    tree = uproot.open(f)[t]
    if n > 0: # if n > 0 n is the number of entries to do training on 
        df    = tree.pandas.df(events, entrystop = n)
        jetdf = tree.pandas.df(jets, entrystop = n)
        lepdf = tree.pandas.df(leps, entrystop = n)
    else: # else do training on the full sample
        df    = tree.pandas.df(events)
        jetdf = tree.pandas.df(jets)
        lepdf = tree.pandas.df(leps)
    if do == "dilepton":
        nJet = 2; nLep = 2
        dfj1 = jetdf.xs(0, level='subentry')
        dfj2 = jetdf.xs(1, level='subentry')
        dfl1 = lepdf.xs(0, level='subentry')
        dfl2 = lepdf.xs(1, level='subentry')
        final = df.assign(Jet1_Pt = dfj1['Jet_Pt'], Jet1_Mass=dfj1['Jet_Mass'], 
                          Jet2_Pt = dfj2['Jet_Pt'], Jet2_Mass=dfj2['Jet_Mass'],                       
                          Lep1_Pt = dfl1['Lepton_Pt'],
                          Lep2_Pt = dfl2['Lepton_Pt']).fillna(0.0)   
    if do == "SingleLepP" or do == "SingleLepM":
        nJet = 3; nLep = 1
        dfj1 = jetdf.xs(0, level='subentry')
        dfj2 = jetdf.xs(1, level='subentry')
        dfj3 = jetdf.xs(2, level='subentry')
        dfl1 = lepdf.xs(0, level='subentry')
        final = df.assign(Jet1_Pt = dfj1['Jet_Pt'], Jet1_Mass=dfj1['Jet_Mass'], 
                          Jet2_Pt = dfj2['Jet_Pt'], Jet2_Mass=dfj2['Jet_Mass'],     
                          Jet3_Pt = dfj3['Jet_Pt'], Jet3_Mass=dfj3['Jet_Mass'],     
                          Lep1_Pt = dfl1['Lepton_Pt']).fillna(0.0)            
    if do == "AllHadronic":
        nJet = 4; nLep = 0
        dfj1 = jetdf.xs(0, level='subentry')
        dfj2 = jetdf.xs(1, level='subentry')
        dfj3 = jetdf.xs(2, level='subentry')
        dfj4 = jetdf.xs(3, level='subentry')
        final = df.assign(Jet1_Pt = dfj1['Jet_Pt'], Jet1_Mass=dfj1['Jet_Mass'], 
                          Jet2_Pt = dfj2['Jet_Pt'], Jet2_Mass=dfj2['Jet_Mass'],   
                          Jet3_Pt = dfj3['Jet_Pt'], Jet3_Mass=dfj3['Jet_Mass'],   
                          Jet4_Pt = dfj4['Jet_Pt'], Jet4_Mass=dfj4['Jet_Mass']).fillna(0.0)          
    labels =  ['Number of jets', '$\mathrm{p_{T}^{miss}}$ [GeV]']
    for j in range(1, nJet+1):
        labels.append('Jet '+str(j)+' $\mathrm{p_{T}}$ [GeV]')
        labels.append('Jet '+str(j)+' mass [GeV]')
    for l in range(1, nLep+1):
        labels.append('Lepton '+str(j)+' $\mathrm{p_{T}}$ [GeV]')

    return final, labels

def create_missing_folders(folders):
    if folders is None:
        return

    for folder in folders:
        if folder is None or folder == "":
            continue

        if not os.path.exists(folder):
            os.makedirs(folder)

        elif not os.path.isdir(folder):
            raise OSError("Path {} exists, but is no directory!".format(folder))


def load_and_check(filename, warning_threshold=1.0e9, memmap_files_larger_than_gb=None):
    if filename is None:
        return None

    if not isinstance(filename, six.string_types):
        data = filename
        memmap = False
    else:
        filesize_gb = os.stat(filename).st_size / 1.0 * 1024 ** 3
        if memmap_files_larger_than_gb is None or filesize_gb <= memmap_files_larger_than_gb:
            logger.info("  Loading %s into RAM", filename)
            data = np.load(filename)
            memmap = False
        else:
            logger.info("  Loading %s as memory map", filename)
            data = np.load(filename, mmap_mode="c")
            memmap = True

    if not memmap:
        n_nans = np.sum(np.isnan(data))
        n_infs = np.sum(np.isinf(data))
        n_finite = np.sum(np.isfinite(data))
        if n_nans + n_infs > 0:
            logger.warning(
                "%s contains %s NaNs and %s Infs, compared to %s finite numbers!", filename, n_nans, n_infs, n_finite
            )

        smallest = np.nanmin(data)
        largest = np.nanmax(data)
        if np.abs(smallest) > warning_threshold or np.abs(largest) > warning_threshold:
            logger.warning("Warning: file %s has some large numbers, rangin from %s to %s", filename, smallest, largest)

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    return data

def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
