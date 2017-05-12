"""
  Train on epithelium data set obtained from:
  http://www.andrewjanowczyk.com/use-case-2-epithelium-segmentation/
"""

from __future__ import print_function

__author__ = 'mjp, May 2017'
__license__ = 'Apache 2.0'


import os, sys, time

import h5py
import numpy as np

np.random.seed(9999)

import keras
from keras import backend as K

sys.path.append('../..')
from cnn_tools import *
from data_tools import *



if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    tile_size = (256, 256)
    
    # load raw data
    with h5py.File('epi.hdf5') as h5:
        X = h5['X'].value.astype(np.float32) / 255.
        Y = h5['Y'].value.astype(np.int32)

    X = X[:, np.newaxis, ...]  # TODO: add back 3 color channels?
    Y = Y[:, np.newaxis, ...]
    Y[Y==255] = 1

    # XXX: this could be selected more carefully
    train = np.arange(25)
    valid = np.arange(25,30)
        
    print('[info]: using Keras version:     %s' % str(keras.__version__))
    print('[info]: using backend:           %s' % str(K._BACKEND))
    print('[info]: tile size:               %s' % str(tile_size))

    # train model
    tic = time.time()
    model = create_unet((1, tile_size[0], tile_size[1]))
    train_model(X[train,...], Y[train,...],
                X[valid,...], Y[valid,...],
                model,
                n_epochs=20, mb_size=30, n_mb_per_epoch=25)

    print('[info]: total time to train model: %0.2f min' % ((time.time() - tic)/60.))
    

