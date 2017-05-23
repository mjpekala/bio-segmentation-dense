"""
  Implements semantic segmentation (dense classification) using Keras.


  Conventions:
   o I implicitly assume 'th' ordering throughout; that is, I expect objects to
     have the shape:

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

import numpy as np

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

from data_tools import *

import sklearn.metrics as skm
import theano




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
    """

    # In some cases, there may be no label associated with a pixel.
    # This is encoded as a "zero-hot" vector in y_true.
    #
    is_pixel_labeled = K.sum(y_true, axis=1)               # for one-hot or zero-hot, this should be 0 or 1
    is_pixel_labeled = is_pixel_labeled.clip(0,1)          # for multi-label case

    # we could zero out estimates associated with unlabeled pixels, but this is
    # not necessary (multiplying by y_true effectively does this)
    #
    #is_pixel_labeled = is_pixel_labeled[:,np.newaxis,:,:]  # enable broadcast
    #y_hat = y_hat * is_pixel_labeled

    # Normally y_hat is coming from a sigmoid (or other "squashing")
    # and therefore never reaches exactly 0 or 1 (so the call to log
    # below is safe).  However, out of paranoia, we call clip() here.
    y_hat = y_hat.clip(1e-9, 1 - 1e-9)

    # the categorical crossentropy loss
    # ** assumes one-hot encoding and sum-to-one along class dimension **
    loss = K.sum(y_true * K.log(y_hat), axis=1)

    if w is not None:
        raise NotImplementedError('asymmetric weighting is a to-be-implemented feature')
        #ce *= w

    #return K.mean(-loss)
    return K.sum(-loss) / K.sum(is_pixel_labeled)



def create_unet(sz, n_classes=2, multi_label=False):
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
    model.compile(optimizer=Adam(lr=1e-3), loss=pixelwise_ace_loss, metrics=['acc'])

    model.name = 'U-Net'
    return model



def train_model(X_train, Y_train, X_valid, Y_valid, model,
                n_epochs=30, n_mb_per_epoch=25, mb_size=30, xform=True):
    """
    Note: these are not epochs in the usual sense, since we randomly sample
    the data set (vs methodically marching through it)                
    """
    assert(X_train.dtype == np.float32)
    
    sz = model.input_shape[-2:]
    score_all = []
    n_classes = model.output_shape[1]

    # show some info about the training data
    print('[train_model]: X_train is ', X_train.shape, X_train.dtype, np.min(X_train), np.max(X_train))
    print('[train_model]: Y_train is ', Y_train.shape, Y_train.dtype, np.min(Y_train), np.max(Y_train))
    print('[train_model]: X_valid is ', X_valid.shape, X_valid.dtype, np.min(X_valid), np.max(X_valid))
    print('[train_model]: Y_valid is ', Y_valid.shape, Y_valid.dtype, np.min(Y_valid), np.max(Y_valid))
    

    for ii in range(n_epochs):
        # run one "epoch"
        print('\n[train_model]: starting "epoch" %d (of %d)' % (ii, n_epochs))
        for jj in print_generator(range(n_mb_per_epoch)):
            Xi, Yi = random_minibatch(X_train, Y_train, mb_size, sz, xform)
            Yi = pixelwise_one_hot(Yi, n_classes)
            loss, acc = model.train_on_batch(Xi, Yi)
            score_all.append(loss)

        # save state
        fn_out = '%s_weights_epoch%04d.hdf5' % (model.name, ii)
        model.save_weights(fn_out)

        # evaluate performance on validation data
        Yi_hat_oh = deploy_model(X_valid, model)  # oh = one-hot
        np.savez('%s_valid_epoch%04d' % (model.name, ii), X=X_valid, Y=Y_valid, Y_hat=Yi_hat_oh, s=score_all)

        Yi_hat = np.argmax(Yi_hat_oh, axis=1);  Yi_hat = Yi_hat[:,np.newaxis,...]
        acc = 100. * np.sum(Yi_hat == Y_valid) / Y_valid.size
        net_prob = np.sum(Yi_hat_oh, axis=1)  # This should be very close to 1 everywhere
        
        print('[train_model]: recent train loss: %0.3f' % np.mean(score_all[-20:]))
        print('[train_model]: acc on validation data:   %0.3f' % acc)
        
        if n_classes == 2 and np.any(Yi_hat > 0):
            # easy to do an f1 score in binary case
            print('[train_model]: f1 on validation data:    %0.3f' % skm.f1_score(Y_valid.flatten(), Yi_hat.flatten()))

        # look at the distribution of class labels
        for ii in range(n_classes):
            frac_ii_yhat = 1. * np.sum(Yi_hat_oh[:,ii,...]) / Y_valid.size # "prob mass" in class ii
            frac_ii_y = 1. * np.sum(Y_valid == ii) / Y_valid.size
            print('[train_model]:    frac y=%d:  %0.3f (%0.3f)' % (ii, frac_ii_yhat, frac_ii_y))

    return score_all



def deploy_model(X, model, two_pass=False):
    """ Runs a deep network on an unlabeled test data

       X    : a tensor of dimensions (n_examples, n_channels, n_rows, n_cols)
      model : a Keras deep network

     Note: the code below uses n_examples as the mini-batch size (so the caller may want
           to invoke this function multiple times for different subsets of slices)
    """

    tile_rows, tile_cols = model.input_shape[-2:]  # "tile" size
    tile_gen = tile_generator(X, [tile_rows, tile_cols])

    Y_hat = None  # delay initialization until we know the # of classes

    # loop over all tiles in the image
    for Xi, (rr,cc) in tile_gen:
        Yi = model.predict(Xi)
        
        # create Y_hat if needed
        if Y_hat is None:
            Y_hat = np.zeros((X.shape[0], Yi.shape[1], X.shape[2], X.shape[3]), dtype=Yi.dtype)

        # store the result
        Y_hat[:,:, rr:(rr+tile_rows), cc:(cc+tile_cols)] = Yi

        
    # optional: do another pass at a different offset (e.g. to clean up edge effects)
    if two_pass:
        tile_gen = tile_generator(X, [tile_rows, tile_cols], offset=[int(tile_rows/2), int(tile_cols/2)])
        for Xi, (rr,cc) in tile_gen:
            Yi = model.predict(Xi)

            # the fraction of the interior to use could perhaps be a parameter.
            frac_r, frac_c = int(tile_rows/10), int(tile_cols/10)
            ra, ca = rr+frac_r, cc+frac_c
            dr, dc = 8*frac_r, 8*frac_c
            
            # store (a subset of) the result
            Y_hat[:, :, ra:(ra+dr), ca:(ca+dc)] = Yi[:, :, frac_r:(frac_r+dr), frac_c:(frac_c+dc)]
        
    return Y_hat

