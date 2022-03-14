from __future__ import absolute_import, division, print_function, unicode_literals

import six
import logging
import os
import torch
from torch.nn import functional as F
from torch import optim
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss
import numpy as np

from contextlib import contextmanager

logger = logging.getLogger(__name__)

def get_activation(activation):
    if activation == "relu":
        return torch.relu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "sigmoid":
        return torch.sigmoid
    elif activation == "log_sigmoid":
        return F.logsigmoid
    else:
        raise ValueError("Activation function %s unknown", activation)


def get_loss(method, alpha, w = 1, loss_type="regular"):
    if method in ["carl", "carl2"]:
        loss_functions = [ratio_xe(loss_type)]
        # It is advised not to use loss_weights inside this function like this
        loss_weights = [1.0] #sjiggins
        loss_labels = ["xe"]
    else:
        raise NotImplementedError("Unknown method {}".format(method))

    return loss_functions, loss_labels, loss_weights #sjiggins


def get_optimizer(optimizer, nesterov_momentum):
    opt_kwargs = None
    if optimizer == "adam":
        opt = optim.Adam
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


def _ratio_xe(s_hat, y_true, w):
    # New weighted loss functions - sjiggins
    epsilon_1 = 0.9
    epsilon_2 = 0.9
    if w is None:
        w = torch.ones(y_true.shape[0])
    #loss = BCELoss(weight=w, reduction='mean')(s_hat, y_true)
    

    ### Version 2
    ##inverse = torch.reciprocal(s_hat)
    ##inverse_sub = torch.reciprocal(1-s_hat)
    ##addTerm = torch.max(inverse, inverse_sub)
    #loss = BCELoss(weight=w, reduction='mean')(s_hat, y_true)
    #prob = torch.pow(s_hat, 2)
    #prob = torch.sub( torch.ones(prob.size()).to("cuda", torch.float, non_blocking=True), prob )
    #prob = torch.reciprocal(prob)
    #prob = torch.exp(prob)
    #loss = torch.add(loss, prob)
    #loss = torch.sum(loss)

    #### Version 1
    #print(loss)
    ##  Test of output addition to loss
    bceloss = BCELoss(weight=w, reduction='none')
    loss = bceloss(s_hat, y_true)
    #print(loss)
    
    #s_hat_temp = torch.mul(s_hat, epsilon_1)#.to("cuda", torch.float, non_blocking=True)
    #s_hat_temp = torch.mul(s_hat, 1.0)#.to("cuda", torch.float, non_blocking=True)
    #s_hat_temp = torch.where(s_hat > 0.9, s_hat, torch.zeros(torch.size(s_hat)))
    #s_hat_temp = 1-s_hat
    #inverse = torch.reciprocal(s_hat_temp)
    #inverse = torch.reciprocal(s_hat)
    #inverse_sub = torch.reciprocal(1-s_hat_temp)


    #### Automatic fraction adaptor per batch ###    
    frac = 0.5
    s_hat_temp = torch.sub(s_hat, 0.5)  #works 3
    s_hat_temp = torch.where(s_hat_temp > 0, s_hat_temp, torch.zeros(s_hat_temp.size()).to("cuda", torch.float, non_blocking=True))
    s_hat_temp = torch.mul(s_hat_temp, 2)       ## works 3
    s_hat_temp = torch.pow(s_hat_temp, 4)            ## Works 3
    inverse_sub = torch.reciprocal(1-s_hat_temp)      ## Works 3
    # Calculate the temporary batch loss from the 
    tempTotal = torch.sum(loss)           ## work 3b
    print("tempTotal: {}".format(tempTotal))
    hatTempTotal = torch.sum(inverse_sub)  ## work 3b
    print("hatTempTotal: {}".format(hatTempTotal))
    #coefficient = 0.1 - (tempTotal/hatTempTotal)
    coefficient = tempTotal/( torch.div(hatTempTotal,frac) - hatTempTotal )
    coefficient=0.1
    print("coefficient: {}".format(coefficient))
    #inverse_sub = torch.mul(inverse_sub, 0.1)      ## Works 3
    inverse_sub = torch.mul(inverse_sub, coefficient)      ## Works 3b

    #s_hat_temp = torch.pow(s_hat, 10)            ## Works 2
    #inverse_sub = torch.reciprocal(1-s_hat)      ## Works 2
    #inverse_sub = torch.mul(inverse_sub, 0.1)      ## Works 2

    #inverse_sub = torch.reciprocal(1-s_hat)      ## Works 1
    #inverse_sub = torch.where(inverse_sub > 10, inverse_sub, torch.zeros(inverse_sub.size()).to("cuda", torch.float, non_blocking=True) ) ### Works 1

    #inverse_sub = torch.reciprocal(1-s_hat)
    #print("s_hat: {}".format(s_hat))
    #print("1/1-s_hat: {}".format(inverse_sub))
    #print("1-s_hat: {}".format(s_hat_temp))
    #addTerm = torch.max(inverse, inverse_sub)
    
    #print("loss shape: {}".format(loss.size()))
    #print("inverse shape: {}".format(inverse.size()))
    #print("inverse_sub shape: {}".format(inverse_sub.size()))
    
    #inverse_sub = torch.mul(inverse_sub, epsilon_2)
    #inverse = torch.mul(inverse, epsilon_1)
    loss = torch.add(loss, inverse_sub) # OutputSuppresion      # Works 1
    #loss = torch.add(loss, inverse)   # OutputSuppresion
    #loss = torch.add(loss, s_hat)   # OutputSuppresion
    #loss = torch.add(loss, s_hat_temp)   # OutputSuppresion
    
    ## Best is s_hat + inverse with epsilon = 0.01

    #loss = torch.add(loss, inverse)   # OutputSuppresion-v2
    #loss = torch.add(loss, s_hat_temp) # OutputSuppresion-v2
    #loss = torch.add(loss, addTerm)
    
    #print(inverse)
    #print(inverse_sub)
    #print(loss)
    
    loss = torch.sum(loss)

    return loss

def _ratio_xe_abs_w(s_hat, y_true, w):
    if w is None:
        w = torch.ones(y_true.shape[0])
    loss = BCELoss(weight=torch.abs(w))(s_hat, y_true)
    return loss

def _ratio_xe_log_abs_w(s_hat, y_true, w):
    if w is None:
        w = torch.ones(y_true.shape[0])
    loss = BCELoss(weight=torh.log(torch.abs(w)))(s_hat, y_true)
    return loss

def ratio_xe(type):
    """
    server as factory for different weight modification.
    """
    preserved_type = {
        "regular": _ratio_xe,
        "abs(w)": _ratio_xe_abs_w,
        "log(abs(w))": _ratio_xe_log_abs_w,
    }
    try:
        return preserved_type[type]
    except KeyError as _error:
        raise KeyError(f"ratio_xe requires one of the type from: {preserved_type}") from _error


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
