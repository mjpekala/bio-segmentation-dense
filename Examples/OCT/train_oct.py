"""
   Trains a dense (per-pixel) classifier on an OCT data set.

   REFERENCES:
    [tie16] Tien et al. Performance evaluation of automated segmentation software 
            on optical coherence tomography volume data. J. Biophotonics, 2016.
            http://onlinelibrary.wiley.com/doi/10.1002/jbio.201500239/full
"""

from __future__ import print_function

__author__ = 'mjp, March 2017'
__license__ = 'Apache 2.0'


import os, sys, time

import numpy as np
from scipy.io import loadmat

np.random.seed(9999) # before importing Keras...
from keras import backend as K

sys.path.append('../..')
from cnn_tools import create_unet, train_model
from data_tools import *



def tian_load_data(mat_file):
    mf = loadmat(mat_file)
    X = mf['volumedata']
    Y1 = mf['O1']
    Y2 = mf['O2']

    # shuffle dimensions
    X = np.transpose(X, [2, 0, 1])
    Y1 = np.transpose(Y1, [2, 0, 1])
    Y2 = np.transpose(Y2, [2, 0, 1])
    
    return X, Y1, Y2



def tian_dense_labels(Y, n_rows):
    """Generates dense (per-pixel) labels from boundaries in [tia16] data set.

    Note this is not the finest segmentation possible; here we are just
    interested in learning the layers identified by the annotators in
    [tie16]; that is, the regions between surfaces 1, 2, 4, 6, and 11.

    Note also there are some extra rows in Y that presumably permitted
    the annotaters to provide finer segmentation in some cases; we
    ignore over these for now.

      Our labels           Tian Semantics
     ------------         ----------------
         0                  above surface 1 or below surface 11
         1                  between surfaces 1 & 2
         2                  between surfaces 2 & 4
         3                  between surfaces 4 & 6
         4                  between surfaces 6 & 11
    """
    n_slices, n_boundaries, n_cols = Y.shape
    assert(n_boundaries == 9)

    Y_dense = np.zeros((n_slices, n_rows, n_cols), dtype=np.int32)

    for s in range(n_slices):
        for col in range(n_cols):
            region_1_2 = np.arange(Y[s,0,col], Y[s,1,col]).astype(np.int32)
            Y_dense[s,region_1_2,col] = 1
            
            region_2_4 = np.arange(Y[s,1,col], Y[s,2,col]).astype(np.int32)
            Y_dense[s,region_2_4,col] = 2
            
            region_4_6 = np.arange(Y[s,2,col], Y[s,4,col]).astype(np.int32)
            Y_dense[s,region_4_6,col] = 3
            
            region_6_11 = np.arange(Y[s,4,col], Y[s,7,col]).astype(np.int32)
            Y_dense[s,region_6_11,col] = 4

    return Y_dense
            
    

if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    tile_size = (256, 256)

    # adjust this as needed for your system
    fn=os.path.expanduser('~/Data/Tian_OCT/jbio201500239-sup-0003-Data-S1.mat')
    
    # load raw data
    X, Y1, Y2 = tian_load_data(fn)

    # for now, we just use one of the truth sets
    Y = Y1
    Y = tian_dense_labels(Y, X.shape[1])

    # add "channel" dimension and change to float32
    X = X[:, np.newaxis, :, :].astype(np.float32)
    Y = Y[:, np.newaxis, :, :].astype(np.float32)

    # we may also want to experiment with a simpler problem
    Y_binary = np.copy(Y)
    Y_binary[Y_binary > 0] = 1

    n_classes = np.sum(np.unique(Y) >= 0)
    print('Y native shape:   ', Y.shape)
    print('class labels:     ', str(np.unique(Y)))
    print('one-hot shape:    ', pixelwise_one_hot(Y).shape)
    for yi in range(n_classes):
        print(' class %d fraction: %0.3f' % (yi, 1.*np.sum(Y==yi)/Y.size))
    print('pct missing:       %0.2f' % (100. * np.sum(Y < 0) / Y.size))

    # split into train/valid/test
    # TODO: k fold CV since we are data limited
    train_slices = np.arange(30)
    valid_slices = np.arange(30,35)
    
    X_train = X[train_slices,...]
    Y_train = Y[train_slices,...]

    X_valid = X[valid_slices,...]
    Y_valid = Y[valid_slices,...]

    print('X train: ', X_train.shape, np.min(X_train), np.max(X_train), X.dtype)
    print('Y train: ', Y_train.shape, np.min(Y_train), np.max(Y_train), Y.dtype)

    # train model
    tic = time.time()
    model = create_unet((1, tile_size[0], tile_size[1]), n_classes)
    train_model(X_train, Y_train, X_valid, Y_valid, model,
                n_epochs=20, mb_size=16, n_mb_per_epoch=25, xform=False)

    print('[info]: total time to train model: %0.2f min' % ((time.time() - tic)/60.))

