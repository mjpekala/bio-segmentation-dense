"""
   Trains a dense (per-pixel) classifier on an OCT data set.
"""

from __future__ import print_function

__author__ = 'mjp, March 2017'
__license__ = 'Apache 2.0'


import os, sys, time

import numpy as np

np.random.seed(9999)

from keras import backend as K

sys.path.append('../..')
from cnn_tools import *
from data_tools import *

import oct



if __name__ == '__main__':
    K.set_image_dim_ordering('th')

    # load raw data
    X, Y = oct.load_oct_sample_data('annotated.mat')
    tile_size = X.shape[1:]

    print(np.unique(Y))

    # split into train and valid
    X_train = X[0,...];  X_train = X_train[np.newaxis, np.newaxis, ...]
    Y_train = Y[0,...];  Y_train = Y_train[np.newaxis, ...]

    X_valid = X[1,...];  X_valid = X_valid[np.newaxis, np.newaxis, ...]
    Y_valid = Y[1,...];  Y_valid = Y_valid[np.newaxis, ...]
 
    # train model
    tic = time.time()
    model = create_unet((1, tile_size[0], tile_size[1]))
    train_model(X_train, Y_train, X_valid, Y_valid, model,
                n_epochs=12, mb_size=1, n_mb_per_epoch=25)

    print('[info]: total time to train model: %0.2f min' % ((time.time() - tic)/60.))
    

