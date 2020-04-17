from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import torch
from torch import tensor

from .models import RatioModel

logger = logging.getLogger(__name__)

def evaluate_ratio_model(
    model,
    method_type=None,
    xs=None,
    run_on_gpu=True,
    double_precision=False,
    return_grad_x=False,
):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Prepare data
    n_xs = len(xs)
    xs = torch.stack([tensor(i) for i in xs])

    model = model.to(device, dtype)
    xs = xs.to(device, dtype)
    with torch.no_grad():
        model.eval()

        r_hat, s_hat= model(xs)
        print("s_hat", s_hat)
        print("r_hat", r_hat)
        # Copy back tensors to CPU
        if run_on_gpu:
            s_hat = s_hat.cpu()
            r_hat = r_hat.cpu()

        # Get data and return
        s_hat = s_hat.detach().numpy().flatten()
        r_hat = r_hat.detach().numpy().flatten()

    return r_hat, s_hat
