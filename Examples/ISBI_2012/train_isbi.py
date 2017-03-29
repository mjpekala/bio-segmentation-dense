"""
   Trains a dense (per-pixel) classifier on the ISBI 2012 data set.

   http://brainiac2.mit.edu/isbi_challenge/
"""

from __future__ import print_function

__author__ = 'mjp, Nov 2016'
__license__ = 'Apache 2.0'


import os, sys, time

import numpy as np

np.random.seed(9999)

from keras import backend as K

sys.path.append('../..')
from cnn_tools import *
from data_tools import *



if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    tile_size = (256, 256)
    
    # load raw data
    isbi_dir = os.path.expanduser('~/Data/ISBI-2012')
    X_train = load_multilayer_tiff(os.path.join(isbi_dir, 'train-volume.tif'))
    Y_train = load_multilayer_tiff(os.path.join(isbi_dir, 'train-labels.tif'))

    X_train = X_train / 255.
    # TODO: normalize image data?
    Y_train = 1 - Y_train / 255.  # map to [0 1] and make 1 := membrane
    Y_train = Y_train.astype(np.int32)

    # split into train and valid
    train_slices = range(20)
    valid_slices = range(25,30)
    X_valid = X_train[valid_slices,...]
    Y_valid = Y_train[valid_slices,...]
    X_train = X_train[train_slices,...]
    Y_train = Y_train[train_slices,...]
 
    print('[info]: training data has shape:     %s' % str(X_train.shape))
    print('[info]: training labels has shape:   %s' % str(Y_train.shape))
    print('[info]: validation data has shape:   %s' % str(X_valid.shape))
    print('[info]: validation labels has shape: %s' % str(Y_valid.shape))
    print('[info]: tile size:                   %s' % str(tile_size))

    # train model
    tic = time.time()
    model = create_unet((1, tile_size[0], tile_size[1]))
    train_model(X_train, Y_train, X_valid, Y_valid, model,
                n_epochs=12, mb_size=30, n_mb_per_epoch=25)

    print('[info]: total time to train model: %0.2f min' % ((time.time() - tic)/60.))
    

