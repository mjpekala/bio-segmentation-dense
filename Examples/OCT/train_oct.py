"""
   Trains a dense (per-pixel) classifier on OCT data sets.

   Initially we focus on the data set of [tia16]; as more data becomes
   available for training, we will switch to using this as a test data
   (to facilitate comparisons with other algorithms that were not
   explicitly trained on [tia16]).

   REFERENCES:
    [tia16] Tian et al. Performance evaluation of automated segmentation software 
            on optical coherence tomography volume data. J. Biophotonics, 2016.
            http://onlinelibrary.wiley.com/doi/10.1002/jbio.201500239/full
"""



from __future__ import print_function

__author__ = 'mjp, March 2017'
__license__ = 'Apache 2.0'


import os, sys, time
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from functools import partial
import h5py

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

np.random.seed(9999) # before importing Keras...
from keras import backend as K
import theano
import matplotlib.pyplot as plt

sys.path.append('../..')
import cnn_tools as ct
import data_tools as dt



TIAN_FILL_ABOVE_CLASS = 0
TIAN_FILL_BELOW_CLASS = 5
TIAN_DONTCARE_CLASS = 6


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
    #
    # UPDATE: changed fold ids based on how the images are laid out!
    #
    n_folds = 5
    fold_id = np.floor(np.arange(X.shape[0]) / n_folds).astype(np.int32)
    
    return X, Y1, Y2, fold_id



def tian_load_wavelet_data(fn):
    """ Loads preprocessed (multi-channel) version of Tian dataset."""
    with h5py.File(fn) as h5:
        X = h5['X_wavelet'].value
        Y1 = h5['Y1'].value
        Y2 = h5['Y2'].value

    # shuffle dimensions
    # I'm a little unclear why rows/cols got swapped ...
    X = np.transpose(X, [0, 1, 3, 2]) 
    Y1 = np.transpose(Y1, [0, 2, 1])
    Y2 = np.transpose(Y2, [0, 2, 1])
    
    # TODO: update if we learn anything about mapping of patients -> images
    # UPDATE: be careful here - it seems the images are in the form:
    #
    #    fovea_1, perifovea1_1, perifovea2_1, parafoveal1_1, parafoveal2_1, fovea_2, ...
    #
    n_folds = 5
    #fold_id = np.mod(np.arange(X.shape[0]), n_folds)
    fold_id = np.floor(np.arange(X.shape[0]) / n_folds)
    
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

    Y_dense = TIAN_FILL_ABOVE_CLASS * np.ones((n_slices, n_rows, n_cols), dtype=np.int32)

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
            Y_dense[s,region_rest,col] = TIAN_FILL_BELOW_CLASS

    return Y_dense



def tian_preprocessing(X, Y, tile_size, mirror_edges=False):
    """ Preprocessing to prepare Tian data for CNN.

      Y should be a set of dense class labels; see tian_dense_labels()
    """

    #----------------------------------------
    # add "channel" dimension (if needed) and change to float32
    #----------------------------------------
    if X.ndim == 3:
        X = X[:, np.newaxis, :, :]
    if Y.ndim == 3:
        Y = Y[:, np.newaxis, :, :]
        
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    #----------------------------------------
    # Pad vertical extent to a power of 2.
    # This is because we want each "tile" to cover the full vertical extent of the image.
    #----------------------------------------
    n_examp, n_chan, n_row, n_col = X.shape
    
    delta_row = tile_size[0] - n_row

    pad_x_shape = (n_examp, n_chan, delta_row, n_col)
    pad_x = np.zeros(pad_x_shape, dtype=X.dtype)
    X = np.concatenate((X, 0*pad_x), axis=-2)

    pad_y_shape = (n_examp, 1, delta_row, n_col)
    pad_y = np.ones(pad_y_shape, dtype=Y.dtype)
    Y = np.concatenate((Y, TIAN_FILL_BELOW_CLASS*pad_y), axis=-2)

    #----------------------------------------
    # some of the borders look bad (missing data but extrapolated labels, etc.).
    # mitigate that here
    #----------------------------------------
    if False:
        # to compensate somewhat, we'll crop away some columns
        # This isn't ideal because:
        #   (a) it also discards legitimate data,
        #   (b) it is not precise, and
        #   (c) it does not address "all zero" rows
        #
        snip_lr = 10
        print('snipping %d columns from both edges!!' % snip_lr)
        X = X[...,snip_lr:-snip_lr]
        Y = Y[...,snip_lr:-snip_lr]

    if True:
        # Suppress class labels from "all zero" columns and/or rows
        # This way, they do not influence the loss function.
        for slice in range(X.shape[0]):
            max_pixel_in_col = np.max(X[slice,0,:,:], axis=0)
            if np.any(max_pixel_in_col==0):
                Y[slice,0,:,max_pixel_in_col==0] = TIAN_DONTCARE_CLASS  

            # rows are a bit tricker, since this interacts with our synthetic
            # data augmentation.
            #
            #max_pixel_in_row = np.max(X[slice,0,:,:], axis=1)
            #if np.any(max_pixel_in_row==0):
            #    Y[slice,0,max_pixel_in_row==0,:] = -1

    #----------------------------------------
    # edge mirroring
    #----------------------------------------
    if mirror_edges:
        X = dt.mirror_edges_lr(X, 100)
        Y = dt.mirror_edges_lr(Y, 100)
        
    return X, Y



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



def tian_shift_updown(X, Y, max_shift=50):
    """ Synthetic data augmentation for Tian data set.

    Randomly shifts the images/labels for a minibatch up or down.

    Note: for Y, must fill top and bottom using different values/labels.

    Note: Y is not yet onehot at this point.
    """

    [n,d,r,c] = X.shape

    delta = np.floor(np.random.rand() * max_shift).astype(np.int32)
    if delta == 0:
        return X, Y
    
    fill_x = np.zeros((n,d,delta,c), dtype=X.dtype)
    fill_y = np.ones((n,1,delta,c), dtype=Y.dtype)
    
    if np.random.rand() < .5:
        X_out = np.concatenate((X[:, :, delta:, :], fill_x), axis=2)
        Y_out = np.concatenate((Y[:, :, delta:, :], TIAN_FILL_BELOW_CLASS*fill_y), axis=2)
    else:
        X_out = np.concatenate((fill_x, X[:, :, :-delta, :]), axis=2)
        Y_out = np.concatenate((TIAN_FILL_ABOVE_CLASS*fill_y, Y[:, :, :-delta, :]), axis=2)

    assert(np.all(X_out.shape == X.shape))
    assert(np.all(Y_out.shape == Y.shape))
    
    return X_out, Y_out
    


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
        

       
 
def ex_smoothness_constraint(X, Y, folds, tile_size, n_epochs=30,
                             layer_weights = [1, 10, 10, 10, 10, 1],
                             ace_tv_weights = [20, .01],
                             out_dir='./Ex_ACE_and_TV'):
    """ Single classifier that encourages smooth estimates.
    """

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    print('layer weights:    %s' % str(layer_weights))
    print('ACE / TV weights: %s' % str(ace_tv_weights))
    
    n_classes = len(np.unique(Y[Y>=0].flatten()))
    all_fold_ids = np.unique(folds).astype(np.int32)
    n_folds = len(all_fold_ids)

    sys.stdout = Tee(os.path.join(out_dir, 'PID%d_logfile.txt' % os.getpid()))
    
    for test_fold in all_fold_ids:
        if test_fold > 0: break # TEMP only run one fold for now while testing
            
        #
        # determine train/valid split for this fold
        #
        avail_folds = [x for x in all_fold_ids if x != test_fold]
        train_folds = avail_folds[:-1]
        valid_fold = avail_folds[-1]
        print('train folds: ', train_folds, ', valid fold(s): ', valid_fold, ', test fold(s): ', test_fold)

        train_slices = [x for x in range(X.shape[0]) if folds[x] in train_folds]
        valid_slices = [x for x in range(X.shape[0]) if folds[x] == valid_fold]
        test_slices  = [x for x in range(X.shape[0]) if folds[x] == test_fold]


        # 
        # custom loss function
        #
        ace_w = partial(ct.pixelwise_ace_loss, w=np.array(layer_weights))
        loss = partial(ct.make_composite_loss,
                           loss_a=ace_w, w_a=ace_tv_weights[0],
                           loss_b=ct.total_variation_loss, w_b=ace_tv_weights[1])
                           # loss_b=ct.l1_smooth_loss, w_b=ace_tv_weights[1])
                           # loss_b=ct.monotonic_in_row_loss, w_b=0.9)
        loss.__name__ = 'custom loss function'  # Keras checks this for something

        #
        # create & train model
        # Note: I reduced the mini-batch size since the tiles are larger now.
        #
        model = ct.create_unet((X.shape[1], tile_size[0], tile_size[1]), n_classes, f_loss=loss)
        model.name = 'PID%d_oct_seg_fold%d' % (os.getpid(), test_fold)
        print('train slices: ', train_slices) # TEMP

        # f_augment = partial(dt.random_minibatch, p_fliplr=.5, f_upstream=tian_shift_updown)
        # more rigorous data augmentation
        f_augment = partial(dt.random_minibatch, p_fliplr=.5, f_upstream=tian_shift_updown, do_random_brightness_adj=True, do_random_blur_or_sharpen=True, do_random_zoom_and_crop=False)

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
        #Y_hat = ct.deploy_model(X, model, two_pass=True)
        Y_hat = [ct.deploy_model(X[ [ii,],...], model, two_pass=True) for ii in range(X.shape[0])]
        Y_hat = np.concatenate(Y_hat, axis=0)
        Y_hat = np.argmax(Y_hat, axis=1)
        acc_test = 100. * np.sum(Y_hat[test_slices,...] == np.squeeze(Y[test_slices,...])) / Y_hat[test_slices,...].size

        C = confusion_matrix(Y[test_slices,...].flatten(), Y_hat[test_slices,...].flatten())
        acc_per_class = 100. * np.diag(C) / np.sum(C,axis=1)

        print('acc test (aggregate): ', acc_test)
        print('acc test (per-class): ', acc_per_class)
        print(C)
        
        fn = '%s_deploy_final' % (model.name)
        fn = os.path.join(out_dir, fn)
        np.savez(fn, X=X, Y=Y, Y_hat=Y_hat, test_slices=test_slices, valid_slices=valid_slices)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
    K.set_image_dim_ordering('th')  # TODO: change to tensorflow ordering!
    tile_size = (512,256)

    if len(sys.argv) > 1:
        out_dir = 'Ex_' + sys.argv[1]
        w1 = float(sys.argv[2])
        w2 = float(sys.argv[3])
        
        layer_weights = [1, w1, w1, w1, w1, 1, 0]
        ace_tv_weights = [w2, 1]
    else:
        out_dir = 'Ex_Default'
        layer_weights = [1, 10, 10, 10, 10, 1, 0]
        ace_tv_weights = [20, .01]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load and preprocess data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if True:
        # load the raw data
        fn=os.path.expanduser('~/Data/Tian_OCT/jbio201500239-sup-0003-Data-S1.mat')
        X, Y1, Y2, fold_id = tian_load_data(fn)

        
    # Choose ground truth
    Y = Y1 
    #Y = np.round((Y1 + Y2) / 2.0)  # mjp: UPDATED

    Y = tian_dense_labels(Y, X.shape[-2])
    X,Y = tian_preprocessing(X, Y, tile_size)

    n_classes = np.sum(np.unique(Y) >= 0)

    print('')
    print('Y native shape:   ', Y.shape)
    print('class labels:     ', str(np.unique(Y)))
    for yi in np.unique(Y):
        print(' class %d fraction: %0.3f' % (yi, 1.*np.sum(Y==yi)/Y.size))
    print('pct missing:       %0.2f' % (100. * np.sum(Y < 0) / Y.size))
    print('X :', X.shape, np.min(X), np.max(X), X.dtype)
    print('folds :', fold_id)
    print('')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run experiment
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ex_smoothness_constraint(X, Y, fold_id, tile_size=tile_size,
                             n_epochs=30,
                             layer_weights=layer_weights,
                             ace_tv_weights=ace_tv_weights,
                             out_dir=out_dir)
