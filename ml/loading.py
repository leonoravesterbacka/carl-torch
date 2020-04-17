from __future__ import absolute_import, division, print_function, unicode_literals

import time
import logging
import numpy as np
import root_numpy
import pandas as pd
import multiprocessing
from functools import partial

from .tools import create_missing_folders, shuffle
#from .analysis import DataAnalyzer

logger = logging.getLogger(__name__)

def load(filename = None, variables = None):
    print("inhere!")
    if filename is None:
        return None
    f    = root_numpy.root2array(filename, branches=variables)
    df   = pd.DataFrame(f,columns=variables)
    return df

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
            If None or larger than 1, MadMiner will use multiprocessing to parallelize the sampling. In this case,
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

        varis = ['VpT','Njets','j1pT', 'j2pT', 'HT','ptmiss', 'l1pT']
        #varis = ['VpT','Njets','j1pT', 'j2pT', 'HT','ptmiss', 'l1pT','Veta','j1eta','j2eta']
        # load sample X0
        x0 = load(filename = '/eos/user/m/mvesterb/data/madgraph/Nominal.root', variables = varis)
        X0 = x0.to_numpy()

        # load sample X1
        x1 = load(filename = '/eos/user/m/mvesterb/data/sherpa/Nominal.root', variables = varis)
        X1 = x1.to_numpy()
        
        # combine
        x = np.vstack([X0, X1])
        y = np.zeros(x.shape[0])
        y[X0.shape[0] :] = 1.0

        # y shape
        y = y.reshape((-1, 1))

        # save data
        if filename is not None and folder is not None:
            np.save(folder + "/x_" + filename + ".npy", x)
            np.save(folder + "/y_" + filename + ".npy", y)

        return x, y
