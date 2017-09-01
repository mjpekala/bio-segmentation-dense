""" Postprocessing semantic segmentation outputs to produce OCT boundary estimates.

  The basic idea is to pass from dense/per-pixel estimates to
  estimates of transitions between regions of each class.  These
  regions have very special structure in the OCT setting that we
  exploit (e.g. smoothness, monotonicity in y dimension).

  Note: if we obtain volumetric data, it will be interesting to try a 2d GP regression.
"""

__author__ = "mjp"
__date__ = 'july 2017'


import time
from itertools import product
import pdb, unittest

import numpy as np
from skimage.morphology import opening
import pylab as plt

import GPy 



def get_class_transitions(Y_hat, y_above):
    """Returns pixel locations where class transitions occur.

        Y_hat    : (R x C) tensor of dense (ie. per-pixel) class estimates.
        y_above  : the class label immediately above the boundary of interest

    """
    assert(Y_hat.ndim == 2)

    Yi = (Y_hat == y_above)
    Yii = (Y_hat == (y_above+1))

    # NOTE: there is potential off-by-one issue here, depending upon
    # how the boundary pixels are defined.  May need to add 1 to rows
    # depending upon this definition.
    transitions = np.logical_and(Yi[:-1,:], Yii[1:,:])
    rows, cols = np.nonzero(transitions)

    idx = np.argsort(cols)
    #return rows, cols
    return np.c_[rows[idx], cols[idx]]  # return as a two-column matrix



def boundary_regression_1d(x_obs, y_obs, x_eval, kernel=None):
    """Simple GP regression for a single 1-dimensional boundary.

        x_obs  : an (n x 1) vector of observations in the x dimension
        y_obs  : an (n x 1) vector of observations in the y dimension

        x_eval : an (m x 1) vector of locations in the x dimension to interpolate
        kernel : the Gaussian process kernel to use, or None for some default.
    """

    assert(x_obs.size == y_obs.size)

    # GPy wants 2d data even for 1d problems
    x_obs = x_obs if x_obs.ndim == 2 else x_obs[:,np.newaxis]
    y_obs = y_obs if y_obs.ndim == 2 else y_obs[:,np.newaxis]
    x_eval = x_eval if x_eval.ndim == 2 else x_eval[:,np.newaxis]

    # de-mean the data
    mu = np.mean(y_obs)
    y_obs = y_obs - mu

    if kernel is None:
        # some default hyper-parameters.  In practice, these should be
        # determined properly (e.g. maximum likelihood or
        # cross-validation)
        kernel = GPy.kern.RBF(input_dim=1, variance=50., lengthscale=20.)
        
    # fit the Gaussian process and evaluate at desired points
    m = GPy.models.GPRegression(x_obs, y_obs, kernel)
    y_mu, y_sigma = m.predict(x_eval)
    
    return y_mu + mu




def estimate_boundary(Y_hat, class_label, f_regress, interp_only=True):
    """Converts dense (per-pixel) estimates into boundary estimates.
 
    This function basically just automates the process of applying a
    regression procedure to one or more images.

        Y_hat       : (Z x R x C) matrix of Z images, each of which is (R x C).
        class_label : the (scalar) class label whose "lower" boundary we are interested in.
        f_regress   : the regression procedure to use.  This function should take three arguments:
                          f_regress(x_obs, y_obs, x_eval)
                      See boundary_regression_1d for an example.

      RETURNS:
         y_est      : a (Z x C) matrix of estimated boundary values (precisely one per column)
                      Note this matrix may contain NaN (no estimate) values if this 
                      function is precluded from extrapolating.
    """
    # if given a single image, expand to a 3d tensor
    if Y_hat.ndim == 2:
        Y_hat = Y_hat[np.newaxis,...]

    # Note: we use nan as a default value here in case (a) values are
    # missing and (b) we are not asked to interpolate them away.  This
    # makes it clear to the caller (and subsequent metrics
    # calculations) that there is no valid data at these locations.
    y_est = np.nan*np.ones((Y_hat.shape[0], Y_hat.shape[2]))

    # Process each slice/image independently.
    # If these images had some spatial correlation, we could do a 2d regression...
    for z in range(Y_hat.shape[0]):
        Yz = Y_hat[z,...]
        M = get_class_transitions(Yz, class_label)
        rows = M[:,0];  cols = M[:,1]

        x_obs, y_obs = cols, rows
        if not interp_only:
            # estimate over the entire support of the image
            x_eval = np.arange(Yz.shape[1]) 
        else:
            # only regress where we have data to interpolate (no extrapolation)
            x_eval = np.arange(np.min(x_obs), np.max(x_obs)+1)
            
        y_hat = f_regress(x_obs, y_obs, x_eval)
        y_est[z,x_eval] = np.squeeze(y_hat)

    return y_est



def fit_gp_hypers_1d(X_train, Y_train, n_samps=50):
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
    hypers = np.random.rand(n_samps,2)
    hypers[:,0] = 10. + (col_max // 2) * hypers[:,0]  # h : lengthscale
    hypers[:,1] = 10. + (col_max // 4) * hypers[:,1]  # sigma: lengthscale
    hypers[0,:] = np.array([20.,50.])  # a value that may be reasonable

    # evaluate different hypers
    best_score = np.inf
    best_values = (None, None)

    for ii in range(n_samps):
        h, sigma = hypers[ii,0], hypers[ii,1]
        kernel = GPy.kern.RBF(input_dim=1, variance=sigma, lengthscale=h)
        
        scores = []
        for k in all_images:
            r_true = Y_train[Y_train[:,2] == k, 0]
            c_true = Y_train[Y_train[:,2] == k, 1]
            r_hat = boundary_regression_1d(X_train[X_train[:,2] == k, 1],
                                           X_train[X_train[:,2] == k, 0],
                                           c_true, kernel=kernel)
            err_l2 = np.sum((r_hat - r_true)**2)**.5
            err_inf = np.max(np.abs(r_hat - r_true))
            #scores.append(err_l2)
            scores.append(err_inf)

        #score = np.median(np.array(scores))
        score = np.sum(np.array(scores))
        if score < best_score:
            best_score = score
            best_values = (h, sigma)
            print('[fit_gp_hypers_1d]: updating best hypers: ', h, sigma, score)

    return best_values


#-------------------------------------------------------------------------------
# This next section is all experimental
#-------------------------------------------------------------------------------


def _find_outliers_via_gp(x_obs, y_obs):
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

    return is_outlier
    




#-------------------------------------------------------------------------------
# Unit testing codes
#-------------------------------------------------------------------------------

class TestPostprocMethods(unittest.TestCase):
    def test_get_class_transitions(self):
        Y = np.zeros((10,10));
        Y[0,:] = 0
        Y[1,:] = 1
        Y[2:4,:] = 2

        rows, cols = get_class_transitions(Y,0)
        self.assertTrue(len(rows) == 10)
        self.assertTrue(np.all(rows == 0))
        
        rows, cols = get_class_transitions(Y,1)
        self.assertTrue(len(rows) == 10)
        self.assertTrue(np.all(rows == 1))
        
        rows, cols = get_class_transitions(Y,2)
        self.assertTrue(len(rows) == 0)


    def test_estimate_boundary(self):
        n = 10
        Y_hat = np.zeros((n,n));
        Y_hat[0,:] = 0
        Y_hat[1,:] = 1
        Y_hat[2:4,:] = 2

        boundary_1 = estimate_boundary(Y_hat, 1, boundary_regression_1d)[0]
        assert(boundary_1.size == n)
        assert(np.all(boundary_1 == 1))

        # test interpolation-only mode
        Y_hat[:,0] = 100
        boundary_1 = estimate_boundary(Y_hat, 1, boundary_regression_1d, interp_only=True)[0]
        assert(boundary_1.size == n)
        assert(np.all(boundary_1[1:] == 1))
        assert(np.isnan(boundary_1[0]))

    
if __name__ == "__main__":
    unittest.main()
