"""
Codes for postprocessing semantic segmentation outputs to produce OCT boundary estimates.

Note: if we get a substantial amount of volumetric data we can
entertain 2d GP regression as well.  
"""

__author__ = "mjp"
__date__ = 'july 2017'


import numpy as np
from skimage.morphology import opening

import GPy 



def get_class_transitions(Y_hat, upper_class, f_preproc=None):
    """Returns pixel locations where class transitions occur.

        Y           : (r x c) tensor of dense (per-pixel) class estimates.
        upper_class : the class label immediately above the boundary of interest
    """

    assert(Y_hat.ndim == 2)

    Yi = (Y_hat == upper_class)
    Yii = (Y_hat == (upper_class+1))

    if f_preproc:
        Yi = f_preproc(Yi)
        Yii = f_preproc(Yii)

    Delta = np.logical_and(Yi, np.roll(Yii, -1, axis=0))
    rows, cols = np.nonzero(Delta)
    return rows, cols



def simple_boundary_regression_1d(x, y, x_eval=np.arange(968)):
    """Simple GP regression for a single 1-dimensional boundary.

    TODO: tune hyperparameters.
    XXX: could do something useful with the GP variance...
    """

    assert(x.size == y.size)

    # GPy wants 2d data even for 1d problems
    x_obs = x if x.ndim == 2 else x[:,np.newaxis]
    y_obs = y if y.ndim == 2 else y[:,np.newaxis]
    x_e = x_eval if x_eval.ndim == 2 else x_eval[:,np.newaxis]

    # make data zero-mean
    mu = np.mean(y)
    y_obs = y_obs - mu

    # fit GP
    kernel = GPy.kern.RBF(input_dim=1, variance=50., lengthscale=20.) # TODO: hyper-parameter tuning!
    m = GPy.models.GPRegression(x_obs, y_obs, kernel)
    
    # TODO: we need some kind of outlier rejection for lower lengthscales

    y_mu, y_sigma = m.predict(x_e)
    return y_mu + mu
