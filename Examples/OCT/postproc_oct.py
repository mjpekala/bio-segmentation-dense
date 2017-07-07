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

        Y           : (R x C) tensor of dense (per-pixel) class estimates.
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



def find_outliers(x_obs, y_obs):
    """ TODO: a more rigorous approach.
    """

    def calc_outlier_metric(x,y):
        kernel = GPy.kern.RBF(input_dim=1, variance=10., lengthscale=50.) 
        m = GPy.models.GPRegression(x, y, kernel)
        y_mu, y_sigma = m.predict(x)
        sigma_dist = np.abs(y - y_mu) / y_sigma
        return sigma_dist
    
    reshape_gpy = lambda v: v if v.ndim == 2 else v[:,np.newaxis]
        
    # GPy wants 2d data even for 1d problems
    n = x_obs.size
    assert(n == y_obs.size)
    x_obs = reshape_gpy(x_obs)
    y_obs = reshape_gpy(y_obs)

    # incrementally remove outliers
    is_outlier = np.zeros((n,), dtype=bool)
    metric = calc_outlier_metric(x_obs, y_obs)
    thresh = 5

    while np.any(metric > thresh):
        is_outlier[np.argmax(metric)] = np.True_
        metric[:] = 0
        metric[~is_outlier] = calc_outlier_metric(x_obs[~is_outlier], y_obs[~is_outlier])
        
    return np.concatenate((x_obs[is_outlier], y_obs[is_outlier]), axis=1)
    
    

def simple_boundary_regression_1d(x, y, x_eval, kern=None, reject_thresh=np.inf):
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
    if kern is None:
        # some default hyper-parameters.  In practice, these should be determined via cross-validation
        kernel = GPy.kern.RBF(input_dim=1, variance=50., lengthscale=20.) 
    m = GPy.models.GPRegression(x_obs, y_obs, kernel)

    # TODO: we need some kind of outlier rejection for lower lengthscales
    # IDEA: fit with a fairly smooth GP then reject points more than N sigma; then, re-fit with a shorter lengthscale??
    
    if np.isfinite(reject_thresh):
        raise ValueError('outlier rejection not yet implemented')

    y_mu, y_sigma = m.predict(x_e)
    return y_mu + mu



def dense_to_boundary(Y_hat, class_label, f_regress=None):
    """ Converts dense (per-pixel) estimates into single boundary estimates for a specified class.

       Y_hat : (Z x R x C) matrix of Z images, each of which is RxC.
    """
    if Y_hat.ndim == 2:
        Y_hat = Y_hat[np.newaxis,...]

    # Note: we use nan as a default value here in case (a) values are
    # missing and (b) we are not asked ti interpolate them away.  This
    # makes it clear to the caller (and subsequent metrics
    # calculations) that there is no valid data at these locations.
    b_est = np.nan*np.ones((Y_hat.shape[0], Y_hat.shape[2]))
    
    for z in range(Y_hat.shape[0]):
        Yz = Y_hat[z,...]
        rows, cols = get_class_transitions(Yz, class_label)

        if f_regress is not None:
            y_hat = f_regress(cols, rows, np.arange(Yz.shape[1]))
            b_est[z,:] = np.squeeze(y_hat)
        else:
            b_est[z,cols] = np.squeeze(rows)

    return b_est
