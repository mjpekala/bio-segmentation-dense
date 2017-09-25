"""
  Implements semantic segmentation (dense classification) using Keras.


  Conventions:

   o I implicitly assume 'th' (or "channels first") ordering
     throughout; that is, I expect objects to have the shape:

           (#_objects, #_channels, #_rows, #_cols)

     Since our examples so far have all been grayscale I typically assume #_channels
     is 1.  For RGB or other multi-channel data, you may need to make some fixes.

   o To support multi-class problems, I expect (per-pixel) class labels to be contiguous
     integers in  0,...,n_classes  and for the class label tensor to have the shape

           (#_objects, 1, #_rows, #_cols)

     Internally we will expand Y into a one-hot encoding 

           (#_objects, #_classes, #_rows, #_cols)

    o I have only tried using this with Theano as the backend.
"""


from __future__ import print_function

__author__ = 'mjp, Nov 2016'
__license__ = 'Apache 2.0'


import time

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.optimizers import Adam, Nadam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from Examples.OCT.densenet import DenseNetFCN

from data_tools import *

import sklearn.metrics as skm
import theano
import tensorflow as tf

FOLD = 9

def print_generator(c, every_n_secs=60*2):
    """ Generator over a collection c that provides progress information (to stdout)
    """
    start_time = time.time()
    last_chatter = 0 #-every_n_secs

    for idx, ci in enumerate(c):
        yield ci
        
        elapsed = time.time() - start_time
        if (elapsed) > last_chatter + every_n_secs:
            last_chatter = elapsed
            print('processed %d items in %0.2f minutes' % (idx+1, elapsed/60.))


def f1_score(y_true, y_hat):
    """ 
    Note: This is designed for *binary* classification problems.
          We implicitly assume y_true, y_hat have shape:

               (#_examples, 2, #_rows, #_cols)

          which arose from a 1-hot encoding the input tensor with values 
          in the set {0,1}. 
    """

    # by convention, take slice 0 to be the negative class and slice 1 to
    # be the positive class.
    y_true_flat = K.flatten(y_true[:,1,:,:])
    y_hat_flat = K.flatten(y_hat[:,1,:,:])

    true_pos = K.sum(y_hat_flat * y_true_flat)
    pred_pos = K.sum(y_hat_flat)
    is_pos = K.sum(y_true_flat)

    precision = true_pos / (pred_pos + 1e-12)
    recall = true_pos / (is_pos + 1e-12)

    # adding epsilon to the denominator here for the all-wrong corner case
    return 2 * precision * recall / (precision + recall + 1e-12) 


def pixelwise_ace_loss(y_true, y_hat, w=None):
    """ Pixel-wise average crossentropy loss (ACE).
    This should work for both binomial and multinomial cases.

        y_true :  True class labels in one-hot encoding; shape is:
                     (#_examples, #_classes, #_rows, #_cols)

        y_hat  :  Estimated class labels; same shape as y_true

        w      :  Either None, or a vector with dimension #_classes
    """

    # In some cases, there may be no label associated with a pixel.
    # This is encoded as a "zero-hot" vector in y_true.
    #
    is_pixel_labeled = K.sum(y_true, axis=1)               # for one-hot or zero-hot, this should be 0 or 1
    is_pixel_labeled = K.clip(is_pixel_labeled, 0, 1)          # for multi-label case

    # Normally y_hat is coming from a sigmoid (or other "squashing")
    # and therefore never reaches exactly 0 or 1 (so the call to log
    # below is safe).  However, out of paranoia, we call clip() here.
    y_hat = K.clip(y_hat, 1e-9, 1 - 1e-9)

    # the categorical crossentropy loss
    #
    # ** NOTE **
    #  This calculation assumes one-hot encoding and sum-to-one along class dimension.
    #  There is no loss associated with places where y_true is 0, so y_hat could be
    #  all 1s and incurr no loss.
    #
    if w is not None:
        w_onehot = w[np.newaxis, :, np.newaxis, np.newaxis] # enable broadcast
        loss = K.sum(y_true * w_onehot * K.log(y_hat), axis=1)
    else:
        loss = K.sum(y_true * K.log(y_hat), axis=1)
        
    #return K.mean(-loss)
    return K.sum(-loss) / K.sum(is_pixel_labeled)


def pixelwise_ace_loss_channels_last(y_true, y_hat, w=None):
    """ Pixel-wise average crossentropy loss (ACE).
    This should work for both binomial and multinomial cases.

        y_true :  True class labels in one-hot encoding; shape is:
                     (#_examples, #_classes, #_rows, #_cols)

        y_hat  :  Estimated class labels; same shape as y_true

        w      :  Either None, or a vector with dimension #_classes
    """

    # In some cases, there may be no label associated with a pixel.
    # This is encoded as a "zero-hot" vector in y_true.
    #
    is_pixel_labeled = K.sum(y_true, axis=3)  # for one-hot or zero-hot, this should be 0 or 1
    is_pixel_labeled = K.clip(is_pixel_labeled, 0, 1)  # for multi-label case

    # Normally y_hat is coming from a sigmoid (or other "squashing")
    # and therefore never reaches exactly 0 or 1 (so the call to log
    # below is safe).  However, out of paranoia, we call clip() here.
    y_hat = K.clip(y_hat, 1e-9, 1 - 1e-9)

    # the categorical crossentropy loss
    #
    # ** NOTE **
    #  This calculation assumes one-hot encoding and sum-to-one along class dimension.
    #  There is no loss associated with places where y_true is 0, so y_hat could be
    #  all 1s and incurr no loss.
    #
    if w is not None:
        w_onehot = w[np.newaxis, np.newaxis, np.newaxis, :]  # enable broadcast
        loss = K.sum(y_true * w_onehot * K.log(y_hat), axis=3)
    else:
        loss = K.sum(y_true * K.log(y_hat), axis=3)

    # return K.mean(-loss)
    return K.sum(-loss) / K.sum(is_pixel_labeled)


def total_variation_loss(y_true, y_hat):
    """
    adapted from: keras/examples/neural_style_transfer.py
    """
    assert K.ndim(y_hat) == 4
    n_rows = y_hat.shape[-2]
    n_cols = y_hat.shape[-1]
    
    # differences along rows and columns
    # note: I assume channels first.
    #
    # note: even though these encodings are one-hot, this calculation should
    #       still be reasonable (perhaps up to a scaling factor)
    a = K.square(y_hat[:, :, :(n_rows-1), :(n_cols-1)] - y_hat[:, :, 1:, :(n_cols-1)])
    b = K.square(y_hat[:, :, :(n_rows-1), :(n_cols-1)] - y_hat[:, :, :(n_rows-1), 1:])
    
    # a no-op involving y_true so that Theano doesn't complain about
    # unused nodes in the computational graph.
    zero = K.sum(0 * K.flatten(y_true))

    return K.sum(K.pow(a + b, 1.25)) + zero


def total_variation_loss_channels_last(y_true, y_hat):
    """
    adapted from: keras/examples/neural_style_transfer.py
    """
    assert K.ndim(y_hat) == 4
    n_rows = y_hat.shape[-3]
    n_cols = y_hat.shape[-2]

    # differences along rows and columns
    # note: I assume channels first.
    #
    # note: even though these encodings are one-hot, this calculation should
    #       still be reasonable (perhaps up to a scaling factor)
    a = K.square(y_hat[:, :(n_rows - 1), :(n_cols - 1), :] - y_hat[:, 1:, :(n_cols - 1), :])
    b = K.square(y_hat[:, :(n_rows - 1), :(n_cols - 1), :] - y_hat[:, :(n_rows - 1), 1:, :])

    # a no-op involving y_true so that Theano doesn't complain about
    # unused nodes in the computational graph.
    zero = K.sum(0 * K.flatten(y_true))

    return K.sum(K.pow(a + b, 1.25)) + zero


def monotonic_in_row_loss(y_true, y_hat):
    """ Encourages class labels to be strictly increasing in the row dimension.
    """
    assert K.ndim(y_hat) == 4
    n_rows = y_hat.shape[-2]
    n_cols = y_hat.shape[-1]

    # convert one-hot into class estimates.
    # XXX: argmax() may not be terribly convenient to push a gradient through...
    y_hat_flat = y_hat.argmax(axis=1, keepdims=False)

    # if class labels are increasing down the row dimension, then we
    # want the first order difference to be non-negative.
    diff = y_hat_flat[:, 1:n_rows, :] - y_hat_flat[:, :(n_rows-1), :]

    # here we need to decide if the magnitude of the difference is relevant.
    diff = K.clip(diff, -np.Inf, 0)
    # diff = K.clip(dff, -1, 0)

    # a no-op involving y_true so that Theano doesn't complain about
    # unused nodes in the computational graph.
    zero = K.sum(0 * y_true.flatten())

    return K.sum(K.square(diff)) + zero


def l1_smooth_loss(y_true, y_pred):
    """Compute L1-smooth loss.

    # Arguments
        y_true: Ground truth bounding boxes,
            tensor of shape (?, num_boxes, 4).
        y_pred: Predicted bounding boxes,
            tensor of shape (?, num_boxes, 4).

    # Returns
        l1_loss: L1-smooth loss, tensor of shape (?, num_boxes).

    # References
        https://arxiv.org/abs/1504.08083
    """
    abs_loss = tf.abs(y_true - y_pred)
    sq_loss = 0.5 * (y_true - y_pred) ** 2
    l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    return tf.reduce_sum(l1_loss, -1)


def make_composite_loss(y_true, y_hat, loss_a, loss_b, w_a, w_b):
    """ Constructs a linear combination of two loss functions."""
    return loss_a(y_true, y_hat) * w_a + loss_b(y_true, y_hat) * w_b


def weighted_pixelwise_crossentropy(class_weights):

    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        return - tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), class_weights))

    return loss


def create_unet(sz, n_classes=2, multi_label=False, f_loss=pixelwise_ace_loss):
    """
      sz : a tuple specifying the input image size in the form:
           (# channels, # rows, # columns)
      
      References:  
        1. Ronneberger et al. "U-Net: Convolutional Networks for Biomedical
           Image Segmentation." 2015. 
        2. https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
    """
    
    assert(len(sz) == 3)
    if not np.all(np.mod(sz[1:], 16) == 0):
        raise ValueError('This network assumes the input image dimensions are multiple of 2^4')

    # NOTES:
    #   o possibly change Deconvolution2D to UpSampling2D
    #   o updated to Keras 2.0.x API (march 2017)
    #
    bm = 'same'
    
    inputs = Input(sz)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding=bm)(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding=bm)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding=bm)(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding=bm)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding=bm)(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding=bm)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding=bm)(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding=bm)(conv4)
    conv4 = Dropout(.5)(conv4) # mjp
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding=bm)(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding=bm)(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Concatenate(axis=1)([up6, conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding=bm)(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding=bm)(conv6)

    up7 = UpSampling2D(size=(2,2))(conv6)
    up7 = Concatenate(axis=1)([up7, conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding=bm)(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding=bm)(conv7)

    up8 = UpSampling2D(size=(2,2))(conv7)
    up8 = Concatenate(axis=1)([up8, conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding=bm)(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding=bm)(conv8)

    up9 = UpSampling2D(size=(2,2))(conv8)
    up9 = Concatenate(axis=1)([up9, conv1])
    conv9 = Conv2D(32, (3, 3), activation='relu', padding=bm)(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding=bm)(conv9)

    # At this point, "channels" becomes "class labels" (one label per channel)
    #
    # mjp: changed layer below for multinomial case
    #conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    def custom_softmax(x):
        # subtracting the max for numerical stability
        e_x = K.exp(x - K.max(x, axis=1, keepdims=True))
        total = K.sum(e_x, axis=1, keepdims=True) + 1e-6
        softmax = e_x / total
        return softmax

    if multi_label:
        raise RuntimeError('this may not be compatible with the ace loss function!')
        conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)
    else:
        conv10 = Conv2D(n_classes, (1, 1))(conv9)
        conv10 = Lambda(custom_softmax, output_shape=(n_classes,sz[1],sz[2]))(conv10)

    model = Model(inputs=inputs, outputs=conv10)

    # mjp: my f1_score is only for binary case; use acc for now
    #model.compile(optimizer=Adam(lr=1e-3), loss=pixelwise_ace_loss, metrics=[f1_score])
    model.compile(optimizer=Adam(lr=1e-3), loss=f_loss, metrics=['acc'])

    model.name = 'U-Net'
    return model


def create_DenseNetFCN(sz, n_classes=2, f_loss=pixelwise_ace_loss):
    """
      sz : a tuple specifying the input image size in the form:
           (# channels, # rows, # columns)

      References:
        1. Ronneberger et al. "U-Net: Convolutional Networks for Biomedical
           Image Segmentation." 2015.
        2. https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
    """

    assert (len(sz) == 3)

    model = DenseNetFCN(input_shape=sz, dropout_rate=0.2, classes=n_classes, activation='custom_softmax')

    # mjp: my f1_score is only for binary case; use acc for now
    # model.compile(optimizer=Adam(lr=1e-3), loss=pixelwise_ace_loss, metrics=[f1_score])
    model.compile(optimizer=Adam(lr=1e-3), loss=f_loss, metrics=['acc'])

    model.name = 'DenseNetFCN'
    return model


def train_model(X_train, Y_train, X_valid, Y_valid, model,
                n_epochs=30, n_mb_per_epoch=25, mb_size=30, f_augment=random_minibatch, out_dir='.', remove_previous_epoch_saves=False):
    """
    Note: these are not epochs in the usual sense, since we randomly sample
    the data set (vs methodically marching through it)                
    """
    assert(X_train.dtype == np.float32)

    sz = model.input_shape[-2:] if K.image_data_format() == "channels_first" else model.input_shape[1:-1]
    n_classes = model.output_shape[1] if K.image_data_format() == "channels_first" else model.output_shape[-1]
    n_missing_valid = np.sum(np.all(Y_valid < 0, axis=1))
    score_all = []
    acc_best = -1
    prev_fn_out = None
    prev_fn = None

    # show some info about the training data
    print('[train_model]: X_train is ', X_train.shape, X_train.dtype, np.min(X_train), np.max(X_train))
    print('[train_model]: Y_train is ', Y_train.shape, Y_train.dtype, np.min(Y_train), np.max(Y_train))
    print('[train_model]: X_valid is ', X_valid.shape, X_valid.dtype, np.min(X_valid), np.max(X_valid))
    print('[train_model]: Y_valid is ', Y_valid.shape, Y_valid.dtype, np.min(Y_valid), np.max(Y_valid))
    print('[train_model]: model input shape:  ', model.input_shape)
    print('[train_model]: model output shape: ', model.output_shape)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for e_idx in range(n_epochs):
        # run one "epoch"
        print('\n[train_model]: starting "epoch" %d (of %d)' % (e_idx, n_epochs))
        for jj in print_generator(range(n_mb_per_epoch)):
            Xi, Yi = f_augment(X_train, Y_train, mb_size, sz)
            Yi = pixelwise_one_hot(Yi, n_classes)

            if K.image_data_format() == "channels_last":
                Xi = np.transpose(Xi, (0, 2, 3, 1))
                Yi = np.transpose(Yi, (0, 2, 3, 1))

            loss, acc = model.train_on_batch(Xi, Yi)
            score_all.append(loss)

        # evaluate performance on validation data
        Y_valid_hat_oh = deploy_model(X_valid, model)  # oh = one-hot

        Y_valid_hat = np.argmax(Y_valid_hat_oh, axis=1)
        Y_valid_hat = Y_valid_hat[:,np.newaxis,...]
        acc = 100. * np.sum(Y_valid_hat == Y_valid) / Y_valid.size
        net_prob = np.sum(Y_valid_hat_oh, axis=1)  # This should be very close to 1 everywhere

        if len(score_all) > 100:
            print('[train_model]: recent train loss: %0.3f' % np.mean(score_all[-90:]))
        else:
            print('[train_model]: recent train loss: %0.3f' % np.mean(score_all))
            
        print('[train_model]: acc on validation data:   %0.3f' % acc)
        print('[train_model]: optimizer iters:          %d' % K.get_value(model.optimizer.iterations))
        
        if n_classes == 2 and np.any(Y_valid_hat > 0):
            # easy to do an f1 score in binary case
            print('[train_model]: f1 on validation data:    %0.3f' % skm.f1_score(Y_valid.flatten(), Yi_hat.flatten()))

        # look at the distribution of class labels
        #for ii in range(n_classes):
        #    frac_ii_yhat = 1. * np.sum(Yi_hat_oh[:,ii,...]) / Y_valid.size # "prob mass" in class ii
        #    frac_ii_y = 1. * np.sum(Y_valid == ii) / Y_valid.size
        #    print('[train_model]:    [y=%d]  est: %0.3f,  true: %0.3f' % (ii, frac_ii_yhat, frac_ii_y))
        #
        #print('[train_model]:    [y=missing]         true: %0.3f' % (n_missing_valid / Y_valid.size))
        print(skm.classification_report(Y_valid.flatten(), Y_valid_hat.flatten()))
        C = skm.confusion_matrix(Y_valid.flatten(), Y_valid_hat.flatten())
        

        # save state when appropriate
        if (acc > acc_best) or (remove_previous_epoch_saves is False and (e_idx == n_epochs-1)):
            fn_out = '%s_weights_epoch%04d.hdf5' % (model.name, e_idx)
            model.save_weights(os.path.join(out_dir, fn_out))
            if remove_previous_epoch_saves and prev_fn_out is not None:
                os.remove(prev_fn_out)
            prev_fn_out = os.path.join(out_dir, fn_out)

            fn = '%s_valid_epoch%04d.npz' % (model.name, e_idx)
            fn = os.path.join(out_dir, fn)
            np.savez(fn, X=X_valid, Y=Y_valid, Y_hat=Y_valid_hat_oh, s=score_all)
            if remove_previous_epoch_saves and prev_fn is not None:
                os.remove(prev_fn)
            prev_fn = fn
            print("[train_model]: accuracy improved from %f to %f" % (acc_best, acc))
            acc_best = acc

    return score_all


def deploy_model(X, model, two_pass=False):
    """ Runs a deep network on an unlabeled test data

       X    : a tensor of dimensions (n_examples, n_channels, n_rows, n_cols)
      model : a Keras deep network

     Note: the code below uses n_examples as the mini-batch size (so the caller may want
           to invoke this function multiple times for different subsets of slices)
    """

    tile_rows, tile_cols = model.input_shape[-2:] if K.image_data_format() == "channels_first" else model.input_shape[1:-1]
    tile_gen = tile_generator(X, [tile_rows, tile_cols])

    Y_hat = None  # delay initialization until we know the # of classes

    # loop over all tiles in the image
    for Xi, (rr,cc) in tile_gen:
        if K.image_data_format() == "channels_last":
            Xi = np.transpose(Xi, (0, 2, 3, 1))

        Yi = model.predict(Xi)

        if K.image_data_format() == "channels_last":
            Yi = np.transpose(Yi, (0, 3, 1, 2))
        
        # create Y_hat if needed
        if Y_hat is None:
            Y_hat = np.zeros((X.shape[0], Yi.shape[1], X.shape[2], X.shape[3]), dtype=Yi.dtype)

        # store the result
        Y_hat[:,:, rr:(rr+tile_rows), cc:(cc+tile_cols)] = Yi

        
    # optional: do another pass at a different offset (e.g. to clean up edge effects)
    if two_pass:
        tile_gen = tile_generator(X, [tile_rows, tile_cols], offset=[int(tile_rows/2), int(tile_cols/2)])
        for Xi, (rr,cc) in tile_gen:
            if K.image_data_format() == "channels_last":
                Xi = np.transpose(Xi, (0, 2, 3, 1))

            Yi = model.predict(Xi)

            if K.image_data_format() == "channels_last":
                Yi = np.transpose(Yi, (0, 3, 1, 2))

            # the fraction of the interior to use could perhaps be a parameter.
            frac_r, frac_c = int(tile_rows/10), int(tile_cols/10)
            ra, ca = rr+frac_r, cc+frac_c
            dr, dc = 8*frac_r, 8*frac_c
            
            # store (a subset of) the result
            Y_hat[:, :, ra:(ra+dr), ca:(ca+dc)] = Yi[:, :, frac_r:(frac_r+dr), frac_c:(frac_c+dc)]
        
    return Y_hat


def ensemble_models(X, Y, model, ensemble_model_weights, fovea_center_arr, save_results=False, display_results=False, do_crop=False, ensemble_model_names=None):
    Y_hat_raw_per_model = []
    for weights in ensemble_model_weights:
        model.load_weights(weights)
        Y_hat_raw = np.squeeze(np.asarray([deploy_model(X[[ii, ], ...], model, two_pass=True) for ii in range(X.shape[0])]))

        if do_crop:
            # crop from fovea center to convert 9mm scan to 6mm scan (of which metrics are calculated off of in Tian paper)
            Y_hat_raw = batch_horiz_crop_from_fovea_center(Y_hat_raw, new_width=644, crop_axis=3,
                                                                     fovea_center_arr=fovea_center_arr)
        Y_hat_raw_per_model.append(Y_hat_raw)
    Y_hat_raw_per_model = np.asarray(Y_hat_raw_per_model)

    if do_crop:
        # do the same crop that was done above (needed to do this after network predictions, since other algos operate on 9mm)
        X = batch_horiz_crop_from_fovea_center(X, new_width=644, crop_axis=3,
                                               fovea_center_arr=fovea_center_arr)
        Y = batch_horiz_crop_from_fovea_center(Y, new_width=644, crop_axis=2,
                                               fovea_center_arr=fovea_center_arr)


    Y_hat_per_model = np.argmax(Y_hat_raw_per_model, axis=2)

    Y_hat_raw_ensemble_mean = np.mean(Y_hat_raw_per_model, axis=0)


    Y_hat_ensemble_mean = np.argmax(Y_hat_raw_ensemble_mean, axis=1)
    Y_hat_ensemble_std = np.std(Y_hat_raw_ensemble_mean, axis=1)

    if save_results:
        np.savez("DenseNetFCN_Y_ensemble_results_kfold/Y_ensemble_results_fold%d.npz" % FOLD, X=X, Y=Y, Y_hat_raw_per_model=Y_hat_raw_per_model,
                 Y_hat_raw_ensemble_mean=Y_hat_raw_ensemble_mean, Y_hat_per_model=Y_hat_per_model,
                 Y_hat_ensemble_mean=Y_hat_ensemble_mean, Y_hat_ensemble_std=Y_hat_ensemble_std)

    if display_results:
        import matplotlib.pyplot as plt

        for example_i in range(len(Y_hat_ensemble_mean)):
            plt.figure()
            plt.title('X (Example %d)' % example_i)
            plt.imshow(X[example_i, 0].astype(np.uint8))
            plt.colorbar()

            for model_i in range(len(Y_hat_per_model)):
                plt.figure()
                if ensemble_model_names is not None:
                    plt.title('%s - Y_hat_ensemble_%d (Example %d)' % (ensemble_model_names[model_i].split('/')[-1], model_i, example_i))
                else:
                    plt.title('Y_hat_ensemble_%d (Example %d)' % (model_i, example_i))
                plt.imshow(Y_hat_per_model[model_i, example_i])
                plt.colorbar()

            plt.figure()
            plt.title('Y_hat_ensemble_mean (Example %d)' % example_i)
            plt.imshow(Y_hat_ensemble_mean[example_i])
            plt.colorbar()

            plt.figure()
            plt.title('Y_hat_ensemble_std (Example %d)' % example_i)
            plt.imshow((Y_hat_ensemble_std[example_i] * 255.0).astype("uint8"))
            plt.colorbar()

            plt.figure()
            plt.title('Y (Example %d)' % example_i)
            plt.imshow(Y[example_i])
            plt.colorbar()

            plt.show()
            plt.close("all")

    return Y_hat_raw_per_model


def batch_horiz_crop_from_fovea_center(X, new_width, crop_axis, fovea_center_arr):
    assert new_width % 2 == 0, "ERROR: new_width must be an even number"
    # have to swap axes to ensure the crop axis is in a known position (index 1)
    if crop_axis != 1:
        X_swap = np.swapaxes(X.copy(), 1, crop_axis)
    new_X_shape = list(X_swap.shape)
    new_X_shape[1] = new_width
    new_X = np.zeros(tuple(new_X_shape), dtype=X.dtype)
    for i in range(len(X)):
        center = fovea_center_arr[i]
        try:
            new_X[i] = X_swap[i, center - new_width/2:center + new_width/2]
        except IndexError:
            print("ERROR: Image width not big enough to accommodate new_width.")
            exit(-1)
    # undo swap axes
    if crop_axis != 1:
        new_X = np.swapaxes(new_X, 1, crop_axis)
    return new_X


def main():
    # Testing ensemble_models()...
    # K.set_image_dim_ordering('th')
    from keras.models import load_model

    tile_size = (512, 256)
    layer_weights = [1, 10, 10, 10, 10, 1, 0]
    ace_tv_weights = [20, .01]
    fovea_center_arr = np.asarray([382] * 5 + [370] * 5 + [394] * 5 + [370] * 5 + [376] * 5 + [372] * 5 + [372] * 5 + [442] * 5 + [390] * 5 + [366] * 5, dtype=int)

    # ensemble_model_weights = [
    #     "/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/AUG3_0/oct_seg_fold0_weights_epoch0388.hdf5",
    #     "/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/AUG3_1/PID11899_oct_seg_fold0_weights_epoch0440.hdf5",
    #     # "/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/AUG3_2/PID22277_oct_seg_fold0_weights_epoch0468.hdf5",
    #     "/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/AUG3_3/PID555_oct_seg_fold0_weights_epoch0433.hdf5",
    #     # "/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/AUG3_4/PID12142_oct_seg_fold0_weights_epoch0482.hdf5",
    #     "/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/AUG1/oct_seg_fold0_weights_epoch0199.hdf5",
    #     "/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/AUG2/oct_seg_fold0_weights_epoch0155.hdf5",
    #     "/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/L1_SMOOTH/oct_seg_fold0_weights_epoch0178.hdf5",
    #     "/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/NO_CHANGES/oct_seg_fold0_weights_epoch0199.hdf5",
    #     "/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/NO_CHANGES_2/oct_seg_fold0_weights_epoch0188.hdf5",
    # ]

    # Load weights for DenseNetFCN results for each fold (no emsemble here)
    ensemble_model_weights = []

    ensemble_model_weights.append(["/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/DenseNetsFCN_test1/PID26979_oct_seg_fold0_weights_epoch0453.hdf5"])
    ensemble_model_weights.append(["/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/DenseNetsFCN_test1/PID26979_oct_seg_fold1_weights_epoch0242.hdf5"])
    ensemble_model_weights.append(["/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/DenseNetsFCN_test1/PID26979_oct_seg_fold2_weights_epoch0363.hdf5"])
    ensemble_model_weights.append(["/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/DenseNetsFCN_test1/PID26979_oct_seg_fold3_weights_epoch0472.hdf5"])
    ensemble_model_weights.append(["/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/DenseNetsFCN_test1/PID26979_oct_seg_fold4_weights_epoch0497.hdf5"])
    ensemble_model_weights.append(["/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/DenseNetsFCN_test1/PID26979_oct_seg_fold5_weights_epoch0472.hdf5"])
    ensemble_model_weights.append(["/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/DenseNetsFCN_test1/PID26979_oct_seg_fold6_weights_epoch0431.hdf5"])
    ensemble_model_weights.append(["/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/DenseNetsFCN_test1/PID26979_oct_seg_fold7_weights_epoch0424.hdf5"])
    ensemble_model_weights.append(["/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/DenseNetsFCN_test1/PID26979_oct_seg_fold8_weights_epoch0452.hdf5"])
    ensemble_model_weights.append(["/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/DenseNetsFCN_test1/PID26979_oct_seg_fold9_weights_epoch0461.hdf5"])
    ensemble_model_weights = ensemble_model_weights[FOLD]

    results_file = []
    results_file.append("/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/K_FOLD_ENSEMBLE/PID1358_oct_seg_fold0_deploy_final.npz")
    results_file.append("/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/K_FOLD_ENSEMBLE/PID739_oct_seg_fold1_deploy_final.npz")
    results_file.append("/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/K_FOLD_ENSEMBLE/PID739_oct_seg_fold2_deploy_final.npz")
    results_file.append("/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/K_FOLD_ENSEMBLE/PID739_oct_seg_fold3_deploy_final.npz")
    results_file.append("/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/K_FOLD_ENSEMBLE/PID1358_oct_seg_fold4_deploy_final.npz")
    results_file.append("/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/K_FOLD_ENSEMBLE/PID1358_oct_seg_fold5_deploy_final.npz")
    results_file.append("/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/K_FOLD_ENSEMBLE/PID1358_oct_seg_fold6_deploy_final.npz")
    results_file.append("/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/K_FOLD_ENSEMBLE/PID1358_oct_seg_fold7_deploy_final.npz")
    results_file.append("/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/K_FOLD_ENSEMBLE/PID1358_oct_seg_fold8_deploy_final.npz")
    results_file.append("/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/K_FOLD_ENSEMBLE/PID1587_oct_seg_fold9_deploy_final.npz")
    results_file = results_file[FOLD]

    # results = np.load("/home/joshinj1/Projects/bio-segmentation-dense/Examples/OCT/Ex_Default/K_FOLD_ENSEMBLE/PID1587_oct_seg_fold2_deploy_final.npz")
    results = np.load(results_file)

    X = results['X'][results['test_slices']]
    Y = np.squeeze(results['Y'][results['test_slices']])

    # n_classes = len(np.unique(Y[Y >= 0].flatten()))
    n_classes = 7
    ace_w = partial(pixelwise_ace_loss, w=np.array(layer_weights))
    loss = partial(make_composite_loss,
                   loss_a=ace_w, w_a=ace_tv_weights[0],
                   loss_b=total_variation_loss, w_b=ace_tv_weights[1])

    model = create_DenseNetFCN((tile_size[0], tile_size[1], X.shape[1]), n_classes,
                                  f_loss=weighted_pixelwise_crossentropy(layer_weights))
    # model = create_unet((X.shape[1], tile_size[0], tile_size[1]), n_classes, f_loss=loss)

    ensemble_models(X, Y, model, ensemble_model_weights, fovea_center_arr[results['test_slices']], save_results=False, display_results=True, do_crop=False, ensemble_model_names=ensemble_model_weights)

if __name__ == '__main__':
    main()
