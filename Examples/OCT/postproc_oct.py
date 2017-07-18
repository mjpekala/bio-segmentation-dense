"""
Codes for postprocessing semantic segmentation outputs to produce OCT boundary estimates.

Note: if we get a substantial amount of volumetric data we can
entertain 2d GP regression as well.  
"""

__author__ = "mjp"
__date__ = 'july 2017'


import time
from itertools import product

import numpy as np
from skimage.morphology import opening
import pylab as plt

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



def find_inliers(x_obs, y_obs):
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

    # Incrementally remove outliers.
    #
    # We do it this way (vs in one pass) since outliers can pull the
    # GP estimate away from inliers.
    is_outlier = np.zeros((n,), dtype=bool)
    metric = calc_outlier_metric(x_obs, y_obs)
    thresh = 5

    while np.any(metric > thresh):
        is_outlier[np.argmax(metric)] = np.True_  # use np.True_ so that ~  works
        metric[:] = 0
        metric[~is_outlier] = calc_outlier_metric(x_obs[~is_outlier], y_obs[~is_outlier])

    return x_obs[~is_outlier], y_obs[~is_outlier]
    



def fit_gp_hypers_1d(X_train, Y_train):
    """Fits Gaussian process hyperparameters 

    X_train : A (M x 3) matrix of estimated OCT boundary points in the form:

                   row_1, column_1, boundary_id_1
                   row_2, column_2, boundary_id_2
                   ...

              where boundary_id is an integer that identifies which image
              the corresponding coordinate is associated with.

    Y_train : A (N x 3) matrix of true OCT boundary points in the same 
              format as X_train.  Note that N will not equal M in general since
              X_train will have missing and/or duplicate points.
    """
    all_images = np.unique(Y_train[:,2])
    col_max = np.max(Y_train[:,1])

    # setup hypers to search over
    #
    n_guesses = 200
    hypers = np.random.rand(n_guesses,2)
    hypers[:,0] = 10. + (col_max // 2) * hypers[:,0]  # h : lengthscale
    hypers[:,1] = 10. + (col_max // 4) * hypers[:,1]  # sigma: lengthscale
    hypers[0,:] = np.array([20.,50.])  # a value that may be reasonable

    # evaluate different hypers
    best_score = np.inf
    best_values = (None, None)

    for ii in range(n_guesses):
        h, sigma = hypers[ii,0], hypers[ii,1]
        kernel = GPy.kern.RBF(input_dim=1, variance=sigma, lengthscale=h)
        
        scores = []
        for k in all_images:
            r_true = Y_train[Y_train[:,2] == k, 0]
            c_true = Y_train[Y_train[:,2] == k, 1]
            r_hat = simple_boundary_regression_1d(X_train[X_train[:,2] == k, 1],
                                                  X_train[X_train[:,2] == k, 0],
                                                  c_true, kernel=kernel)
            err_l2 = np.sum((r_hat - r_true)**2)**.5
            err_inf = np.max(np.abs(r_hat - r_true))
            scores.append(err_l2)

        score = np.median(np.array(scores))
        if score < best_score:
            best_score = score
            best_values = (h, sigma)
            print(h, sigma, score)

    return best_values


def simple_boundary_regression_1d(x, y, x_eval, kernel=None):
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
    if kernel is None:
        # some default hyper-parameters.  In practice, these should be determined via cross-validation
        kernel = GPy.kern.RBF(input_dim=1, variance=50., lengthscale=20.) 
    m = GPy.models.GPRegression(x_obs, y_obs, kernel)

    # TODO: we need some kind of outlier rejection for lower lengthscales
    # IDEA: fit with a fairly smooth GP then reject points more than N sigma; then, re-fit with a shorter lengthscale??
    
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
            #x_obs, y_obs = find_inliers(cols, rows)
            x_obs, y_obs = cols, rows
            y_hat = f_regress(x_obs, y_obs, np.arange(Yz.shape[1]))
            b_est[z,:] = np.squeeze(y_hat)
        else:
            b_est[z,cols] = np.squeeze(rows)

    return b_est
