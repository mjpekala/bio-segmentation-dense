"""
   Trains a dense (per-pixel) classifier on an OCT data set.

   REFERENCES:
    [tia16] Tian et al. Performance evaluation of automated segmentation software 
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
from cnn_tools import create_unet, train_model, deploy_model
from data_tools import *



def tian_load_data(mat_file):
    """ Loads data set from [tia16]. """
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



def tian_find_crops(Y_est, crop_pct):
    """Chooses vertical crops based on estimated support of layers.

    Our goal here is to reduce the vertical extent of the data set since
    there is a lot of "empty space" contributing to class imbalance.
    This function uses an estimate of the support of the layers of
    interest to pick a number of rows from each slice to retain.

    We'll pick the same # of rows to discard from every slice a-priori
    in order to keep the tensor data structure intact later on.

       Y_est : a tensor with dimensions (#_slices, #_rows, #_cols)
    """
    assert(0 < crop_pct and crop_pct < 1)

    n_slic, n_rows, n_cols = Y_est.shape  

    n_rows_to_keep = int(np.ceil(crop_pct * n_rows))
    box_filt = np.ones((n_rows_to_keep,))

    crops = np.zeros((n_slic, 2))

    for s in range(n_slic):
        # there may be better ways; for now, we convolve the marginal
        # sum of the estimated layer support with a uniform filter to
        # find the best set of rows to keep.
        marginal = np.sum(Y_est[s,:,:], axis=1)
        response = np.convolve(marginal, box_filt, 'same')

        a = np.argmax(response) - n_rows_to_keep / 2
        b = a + n_rows_to_keep

        if b > n_rows:
            delta = b - n_rows
            a -= delta; b -= delta
        elif a < 0:
            delta = 0 - a
            a += delta; b += delta

        crops[s,:] = np.array([a,b])
 
    return crops



def _crop_rows(X, crops):
    n_rows = crops[0,1] - crops[0,0]

    X_out = []
    for s in range(X.shape[0]):
        rows_to_keep = np.arange(crops[s,0], crops[s,1]).astype(np.int32)
        if X.ndim == 4:
            Xs = X[s, :, rows_to_keep, :]
        else:
            Xs = X[s, rows_to_keep, :]
        Xs = Xs[np.newaxis, ...]
        X_out.append(Xs)
                            
    return np.concatenate(X_out, axis=0)



if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    tile_size = (256, 256)
    n_folds = 5

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load and preprocess data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # adjust this as needed for your system
    fn=os.path.expanduser('~/Data/Tian_OCT/jbio201500239-sup-0003-Data-S1.mat')
    X, Y1, Y2 = tian_load_data(fn)

    # for now, we just use one of the truth sets
    Y = Y1
    Y = tian_dense_labels(Y, X.shape[1])

    # add "channel" dimension and change to float32
    X = X[:, np.newaxis, :, :].astype(np.float32)
    Y = Y[:, np.newaxis, :, :].astype(np.float32)

    # class labels for a "layer detection" problem
    Y_binary = np.copy(Y)
    Y_binary[Y_binary > 0] = 1

    # assign data to folds.
    # update if we learn anything about mapping of patients -> images
    fold_id = np.mod(np.arange(X.shape[0]), n_folds)

    n_classes = np.sum(np.unique(Y) >= 0)
    print('Y native shape:   ', Y.shape)
    print('class labels:     ', str(np.unique(Y)))
    for yi in np.unique(Y):
        print(' class %d fraction: %0.3f' % (yi, 1.*np.sum(Y==yi)/Y.size))
    print('pct missing:       %0.2f' % (100. * np.sum(Y < 0) / Y.size))
    print('X :', X.shape, np.min(X), np.max(X), X.dtype)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # run some experiments
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for test_fold in range(n_folds):
        #
        # determine train/valid split for this fold
        #
        avail_folds = [x for x in range(n_folds) if x != test_fold]
        train_folds = avail_folds[:-2]
        valid_fold = avail_folds[-1]

        train_slices = [x for x in range(X.shape[0]) if fold_id[x] in train_folds]
        valid_slices = [x for x in range(X.shape[0]) if fold_id[x] == valid_fold]
        test_slices  = [x for x in range(X.shape[0]) if fold_id[x] == test_fold]

        #
        # train and deploy a model for the "layer detection" problem
        #
        model = create_unet((1, tile_size[0], tile_size[1]), 2)
        model.name = 'oct_detection_fold%d' % test_fold
        
        tic = time.time()
        train_model(X[train_slices,...], Y_binary[train_slices,...],
                    X[valid_slices,...], Y_binary[valid_slices,...],
                    model, n_epochs=20, mb_size=16, n_mb_per_epoch=25, xform=False)
        print('[info]: time to train segmentation model: %0.2f min' % ((time.time() - tic)/60.))

        tv_slices = [x for x in range(X.shape[0]) if fold_id[x] in train_folds + [valid_fold,]]
        Y_hat_layers = deploy_model(X[tv_slices,...], model)
        Y_hat_layers = Y_hat_layers[:,1,:,:] # keep only the postive class estimate

        #
        # create a new data set for the layer estimation problem
        #
        crops = tian_find_crops(Y_hat_layers, .33)
        X_train_s = _crop_rows(X[train_slices, ...], crops)
        Y_train_s = _crop_rows(Y[train_slices, ...], crops)
        X_valid_s = _crop_rows(X[valid_slices, ...], crops)
        Y_valid_s = _crop_rows(Y[valid_slices, ...], crops)
        
        model_s = create_unet((1, 128, 128), n_classes)  # note tile size change
        model_s.name = 'oct_segment_fold%d' % test_fold
        
        tic = time.time()
        train_model(X_train_s, Y_train_s, X_valid_s, Y_valid_s, 
                    model_s, n_epochs=20, mb_size=16, n_mb_per_epoch=25, xform=False)
        print('[info]: total time to train model: %0.2f min' % ((time.time() - tic)/60.))
