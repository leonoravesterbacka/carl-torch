from __future__ import absolute_import, division, print_function, unicode_literals

import six
import logging
import os
import stat
from subprocess import Popen, PIPE
import io
import numpy as np
import shutil
import uproot
import root_numpy
import pandas as pd
import torch
from torch.nn import functional as F

from contextlib import contextmanager

logger = logging.getLogger(__name__)

initialized = False

def load(filename = None, variables = None, n = 0, tree = None):
    if filename is None:
        return None

    f = uproot.open(filename)[tree]
    if n > 0: # if n > 0 n is the number of entries to do training on 
        df = f.pandas.df(variables, entrystop = n)
    else: # else do training on the full sample
        df = f.pandas.df(variables)
    dfj1 =  df.xs(0, level='subentry')
    dfj2 =  df.xs(1, level='subentry')
    
    dfnew = dfj1.assign(Jet1_Pt=  dfj1['Jet_Pt'], Jet1_Eta= dfj1['Jet_Eta'], Jet1_Mass=dfj1['Jet_Mass'], Jet1_Phi= dfj1['Jet_Phi'], 
                        Jet2_Pt=  dfj2['Jet_Pt'], Jet2_Eta= dfj2['Jet_Eta'], Jet2_Mass=dfj2['Jet_Mass'], Jet2_Phi= dfj2['Jet_Phi'],
    )
    final = dfnew.drop(['Jet_Pt', 'Jet_Eta', 'Jet_Mass', 'Jet_Phi'], axis=1).fillna(0.0)
    return final

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
