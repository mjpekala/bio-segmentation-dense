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
from cnn_tools import create_unet, train_model
from data_tools import *

import oct



if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    tile_size = (256, 256)

    # load raw data
    X, Y = oct.load_oct_sample_data('annotated.mat')
    Y = Y[:, np.newaxis, :, :]  # add a "channel" dimension

    # (optional) "binarize" the data for a two-class problem
    if False:
        Y = (Y > 0).astype(np.uint32)

    n_classes = len(np.unique(Y))
    print('Y native shape:   ', Y.shape)
    print('class labels:     ', str(np.unique(Y)))
    print('one-hot shape:    ', pixelwise_one_hot(Y).shape)
    for yi in range(n_classes):
        print(' class %d fraction: %0.3f' % (yi, 1.*np.sum(Y==yi)/Y.size))

    # XXX: also, discard the edges for now
    #      (unclear if annotations extend to edges
    X = X[...,20:-20]
    Y = Y[...,20:-20]

    # split into train and valid.
    # obviously we would prefer more data in the future...
    X_train = X[0,...];  X_train = X_train[np.newaxis, ...]
    Y_train = Y[0,...];  Y_train = Y_train[np.newaxis, ...]

    X_valid = X[1,...];  X_valid = X_valid[np.newaxis, ...]
    Y_valid = Y[1,...];  Y_valid = Y_valid[np.newaxis, ...]

    print('X train: ', X_train.shape, '%0.2g' % np.min(X_train), '%0.2g' % np.max(X_train), X.dtype)
    print('Y train: ', Y_train.shape, np.min(Y_train), np.max(Y_train), Y.dtype)
 
    # train model
    tic = time.time()
    model = create_unet((1, tile_size[0], tile_size[1]), n_classes)
    train_model(X_train, Y_train, X_valid, Y_valid, model,
                n_epochs=20, mb_size=16, n_mb_per_epoch=25, xform=False)

    print('[info]: total time to train model: %0.2f min' % ((time.time() - tic)/60.))

