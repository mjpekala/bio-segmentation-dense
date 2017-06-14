"""
   Trains a dense (per-pixel) classifier on an OCT data set.

   REFERENCES:
    [tia16] Tian et al. Performance evaluation of automated segmentation software 
            on optical coherence tomography volume data. J. Biophotonics, 2016.
            http://onlinelibrary.wiley.com/doi/10.1002/jbio.201500239/full
"""

# TODO: mirror edges and more disciplined sampling of space during training!
# TODO: TV mod p penalizer??


from __future__ import print_function

__author__ = 'mjp, March 2017'
__license__ = 'Apache 2.0'


import os, sys, time
from functools import partial

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

np.random.seed(9999) # before importing Keras...
from keras import backend as K
import theano

sys.path.append('../..')
import cnn_tools as ct
import data_tools as dt



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

    # assign slices to folds
    # TODO: update if we learn anything about mapping of patients -> images
    n_folds = 5
    fold_id = np.mod(np.arange(X.shape[0]), n_folds)
    
    return X, Y1, Y2, fold_id



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
         0                  above surface 1
         1                  between surfaces 1 & 2
         2                  between surfaces 2 & 4
         3                  between surfaces 4 & 6
         4                  between surfaces 6 & 11
         5                  below surface 11
    """
    n_slices, n_boundaries, n_cols = Y.shape
    assert(n_boundaries == 9)

    # make sure values are all integers
    Y = np.round(Y)

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

            region_rest = np.arange(Y[s,7,col], n_rows).astype(np.int32)
            Y_dense[s,region_rest,col] = 5

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
        Y_est_s = Y_est[s,:,:]
        
        # there may be better ways; for now, we convolve the marginal
        # sum of the estimated layer support with a uniform filter to
        # find the best set of rows to keep.
        marginal = np.sum(Y_est_s, axis=1)
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
        Xs = X[[s],...]  # use [.] to dodge the squeeze
        Xs = Xs[..., rows_to_keep, :]
        X_out.append(Xs)
                            
    return np.concatenate(X_out, axis=0)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The following functions run various experiments
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Tee(object):
    def __init__(self, fn='logfile.txt'):
        self.stdout = sys.stdout
        self.logfile = open(fn, 'a')

    def write(self, message):
        self.stdout.write(message)
        self.logfile.write(message)

    def flush(self):
        pass # for python 3, evidently
        

        
def ex_detect_then_segment(X, Y, folds, tile_size, n_epochs=25, out_dir='./Ex_Detect_and_Segment'):

    # class labels for a "layer detection" problem
    Y_binary = np.copy(Y)
    Y_binary[Y_binary > 0 and Y_binary < 5] = 1  # see tien_dense_labels()

    
    for test_fold in range(n_folds):
        if test_fold > 0: break # TEMP TEMP TEMP!! just for quick testing!
            
        #
        # determine train/valid split for this fold
        #
        avail_folds = [x for x in range(n_folds) if x != test_fold]
        train_folds = avail_folds[:-1]
        valid_fold = avail_folds[-1]
        print('train folds: ', train_folds, ', valid fold(s): ', valid_fold, ', test fold(s): ', test_fold)

        train_slices = [x for x in range(X.shape[0]) if fold_id[x] in train_folds]
        valid_slices = [x for x in range(X.shape[0]) if fold_id[x] == valid_fold]
        test_slices  = [x for x in range(X.shape[0]) if fold_id[x] == test_fold]

        #
        # train and deploy a model for the "layer detection" problem
        #
        model = ct.create_unet((1, tile_size[0], tile_size[1]), 2)
        model.name = 'oct_detection_fold%d' % test_fold
        
        tic = time.time()
        ct.train_model(X[train_slices,...], Y_binary[train_slices,...],
                       X[valid_slices,...], Y_binary[valid_slices,...],
                       model, n_epochs=n_epochs, mb_size=16, n_mb_per_epoch=25, xform=False,
                       out_dir=out_dir)
        
        print('[info]: time to train "detection" model: %0.2f min' % ((time.time() - tic)/60.))

        tv_slices = [x for x in range(X.shape[0]) if fold_id[x] in train_folds + [valid_fold,]]
        Y_hat_layers = ct.deploy_model(X[tv_slices,...], model)
        Y_hat_layers = Y_hat_layers[:,1,:,:] # keep only the postive class estimate

        #
        # augment data set for the layer segmentation problem
        #
        crops = tian_find_crops(Y_hat_layers, .33)
        X_train_s = _crop_rows(X[train_slices, ...], crops)
        Y_train_s = _crop_rows(Y[train_slices, ...], crops)
        X_valid_s = _crop_rows(X[valid_slices, ...], crops)
        Y_valid_s = _crop_rows(Y[valid_slices, ...], crops)

        model_s = ct.create_unet((1, 128, 128), n_classes)  # note tile size change
        model_s.name = 'oct_segmentation_fold%d' % test_fold
 
        tic = time.time()
        ct.train_model(X_train_s, Y_train_s, X_valid_s, Y_valid_s, 
                       model_s, n_epochs=n_epochs, mb_size=16, n_mb_per_epoch=25, xform=False,
                       out_dir=out_dir)
        print('[info]: time to train "segmentation" model: %0.2f min' % ((time.time() - tic)/60.))

        #
        # Deploy on test data
        #

        # TODO: some combo of model and model_s may be needed??
        Y_hat_s = ct.deploy_model(X, model_s)
        Y_hat_s = np.argmax(Y_hat_s, axis=1)
        acc_test = 100. * np.sum(Y_hat_s[test_slices,...] == np.squeeze(Y[test_slices,...])) / Y_hat_s[test_slices,...].size

        C = confusion_matrix(Y[test_slices,...].flatten(), Y_hat_s[test_slices,...].flatten())
        acc_per_class = 100. * np.diag(C) / np.sum(C,axis=1)

        print('acc test (aggregate): ', acc_test)
        print('acc test (per-class): ', acc_per_class)
        print(C)


        
def ex_monotonic_loss(X, Y, folds, tile_size, n_epochs=100, out_dir='./Ex_Mono_Labels'):
    """ Single classifier that penalizes out-of-order class labels along vertical dimension.
    """

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    n_classes = len(np.unique(Y.flatten()))
    sys.stdout = Tee(os.path.join(out_dir, 'logfile.txt'))
    
    for test_fold in range(n_folds):
        if test_fold > 0: break # TEMP only run one fold for now while testing
            
        #
        # determine train/valid split for this fold
        #
        avail_folds = [x for x in range(n_folds) if x != test_fold]
        train_folds = avail_folds[:-1]
        valid_fold = avail_folds[-1]
        print('train folds: ', train_folds, ', valid fold(s): ', valid_fold, ', test fold(s): ', test_fold)

        train_slices = [x for x in range(X.shape[0]) if fold_id[x] in train_folds]
        valid_slices = [x for x in range(X.shape[0]) if fold_id[x] == valid_fold]
        test_slices  = [x for x in range(X.shape[0]) if fold_id[x] == test_fold]

        # 
        # custom loss function
        #
        loss = partial(ct.make_composite_loss,
                           loss_a=ct.pixelwise_ace_loss, w_a=0.4,
                           loss_b=ct.monotonic_in_row_loss, w_b=.6)
        loss.__name__ = 'custom loss function'  # Keras checks this for something

        #
        # create & train model
        # Note: I reduced the mini-batch size since the tiles are larger now.
        #
        model = ct.create_unet((1, tile_size[0], tile_size[1]), n_classes, f_loss=loss)
        model.name = 'oct_seg_fold%d' % test_fold

        f_augment = partial(dt.random_minibatch, p_fliplr=.5)

        tic = time.time()
        ct.train_model(X[train_slices,...], Y[train_slices,...],
                       X[valid_slices,...], Y[valid_slices,...],
                       model, n_epochs=n_epochs, mb_size=2, n_mb_per_epoch=25,
                       f_augment=f_augment,
                       out_dir=out_dir)
        
        print('[info]: time to train model: %0.2f min' % ((time.time() - tic)/60.))

        #
        # Deploy
        # Note: we evaluate the whole volume but evaluate performance only on the test subset.
        # Note: we evaluate the volume one slice at a time to avoid memory issues.
        #
        Y_hat = [ct.deploy_model(X[ [ii,],...], model, two_pass=True) for ii in range(X.shape[0])]
        Y_hat = np.concatenate(Y_hat, axis=0)
        #Y_hat = ct.deploy_model(X, model, two_pass=True)
        Y_hat = np.argmax(Y_hat, axis=1)
        acc_test = 100. * np.sum(Y_hat[test_slices,...] == np.squeeze(Y[test_slices,...])) / Y_hat[test_slices,...].size

        C = confusion_matrix(Y[test_slices,...].flatten(), Y_hat[test_slices,...].flatten())
        acc_per_class = 100. * np.diag(C) / np.sum(C,axis=1)

        print('acc test (aggregate): ', acc_test)
        print('acc test (per-class): ', acc_per_class)
        print(C)
        
        fn = '%s_deploy_final' % (model.name)
        fn = os.path.join(out_dir, fn)
        np.savez(fn, X=X, Y=Y, Y_hat=Y_hat, test_slices=test_slices)



        
if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    n_folds = 5
    tile_size = (512,256)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load and preprocess data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # adjust this as needed for your system
    fn=os.path.expanduser('~/Data/Tian_OCT/jbio201500239-sup-0003-Data-S1.mat')
    X, Y1, Y2, fold_id = tian_load_data(fn)

    # for now, we just use one set of annotations 
    Y = Y1
    Y = tian_dense_labels(Y, X.shape[1])

    # pad vertical extent to a power of 2
    delta_row = tile_size[0] - X.shape[1]
    pad = np.ones((X.shape[0], delta_row, X.shape[2]), dtype=X.dtype)
    X = np.concatenate((X, 0*pad), axis=1)
    Y = np.concatenate((Y, 5*pad), axis=1)

    # add "channel" dimension and change to float32
    X = X[:, np.newaxis, :, :].astype(np.float32)
    Y = Y[:, np.newaxis, :, :].astype(np.float32)

    # some of the borders look bad (missing data but extrapolated labels, etc.).
    # to compensate somewhat, we'll crop away some columns
    if True:
        snip_lr = 10
        print('snipping %d columns from both edges!!' % snip_lr)
        X = X[...,snip_lr:-snip_lr]
        Y = Y[...,snip_lr:-snip_lr]

    n_classes = np.sum(np.unique(Y) >= 0)
    
    print('Y native shape:   ', Y.shape)
    print('class labels:     ', str(np.unique(Y)))
    for yi in np.unique(Y):
        print(' class %d fraction: %0.3f' % (yi, 1.*np.sum(Y==yi)/Y.size))
    print('pct missing:       %0.2f' % (100. * np.sum(Y < 0) / Y.size))
    print('X :', X.shape, np.min(X), np.max(X), X.dtype)

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run some experiment
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if False:
        ex_detect_then_segment(X, Y, fold_id)
    else:
        ex_monotonic_loss(X, Y, fold_id, tile_size=tile_size)
