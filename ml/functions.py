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
from torch import optim
from torch.nn import BCELoss, MSELoss

from contextlib import contextmanager

logger = logging.getLogger(__name__)

def get_activation(activation):
    if activation == "relu":
        return torch.relu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "sigmoid":
        return torch.relu
    elif activation == "log_sigmoid":
        return F.logsigmoid
    else:
        raise ValueError("Activation function %s unknown", activation)


def get_loss(method, alpha):
    if method in ["carl", "carl2"]:
        loss_functions = [ratio_xe]
        loss_weights = [1.0]
        loss_labels = ["xe"]
    else:
        raise NotImplementedError("Unknown method {}".format(method))
    return loss_functions, loss_labels, loss_weights

def get_optimizer(optimizer, nesterov_momentum):
    opt_kwargs = None
    if optimizer == "adam":
        opt = optim.Adam
    elif optimizer == "lbfgs":
        opt = optim.LBFGS
    elif optimizer == "amsgrad":
        opt = optim.Adam
        opt_kwargs = {"amsgrad": True}
    elif optimizer == "sgd":
        opt = optim.SGD
        if nesterov_momentum is not None:
            opt_kwargs = {"momentum": nesterov_momentum}
    else:
        raise ValueError("Unknown optimizer {}".format(optimizer))
    return opt, opt_kwargs


def ratio_xe(r_hat, s_hat, y_true):
    loss = BCELoss()(s_hat, y_true)
    return loss

@contextmanager
def less_logging():
    """
    Silences INFO logging messages. Based on https://gist.github.com/simon-weber/7853144
    """

    if logging.root.manager.disable != logging.DEBUG:
        yield
        return

    try:
        logging.disable(logging.INFO)
        yield
    finally:
        logging.disable(logging.DEBUG)                                                       
