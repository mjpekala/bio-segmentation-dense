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
    last_chatter = -every_n_secs

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
    # On these pixels, we do not want to penalize the classifier
    # for making a prediction. Our approach here is to make y_hat
    # artifically all zero on these pixels.
    #
    is_pixel_labeled = K.sum(y_true, axis=1)               # for one-hot or zero-hot, this should be 0 or 1
    is_pixel_labeled = is_pixel_labeled.clip(0,1)          # for multi-label case
    is_pixel_labeled = is_pixel_labeled[:,np.newaxis,:,:]  # enable broadcast
    y_hat = y_hat * is_pixel_labeled

    # Normally y_hat is coming from a sigmoid (or other "squashing")
    # and therefore never reaches exactly 0 or 1 (so the call to log
    # below is safe).  However, if we set to 0 some values of y_hat
    # above, there will be blood.  Hence the call to clip().
    y_hat = y_hat.clip(1e-6, 1 - 1e-6)

    # the categorical crossentropy loss loss
    # elements from this first step live in [-inf,0]
    loss = y_true * K.log(y_hat) + (1. - y_true) * K.log(1. - y_hat)

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
    #up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding=bm)(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding=bm)(conv6)

    up7 = UpSampling2D(size=(2,2))(conv6)
    up7 = Concatenate(axis=1)([up7, conv3])
    #up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding=bm)(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding=bm)(conv7)

    up8 = UpSampling2D(size=(2,2))(conv7)
    up8 = Concatenate(axis=1)([up8, conv2])
    #up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding=bm)(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding=bm)(conv8)

    up9 = UpSampling2D(size=(2,2))(conv8)
    up9 = Concatenate(axis=1)([up9, conv1])
    #up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
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
        conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)
    else:
        conv10 = Conv2D(n_classes, (1, 1))(conv9)
        conv10 = Lambda(custom_softmax, output_shape=(n_classes,sz[1],sz[2]))(conv10)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-3), loss=pixelwise_ace_loss, metrics=[f1_score])

    return model



def train_model(X_train, Y_train, X_valid, Y_valid, model,
                n_epochs=30, n_mb_per_epoch=25, mb_size=30, xform=True):
    """
    Note: these are not epochs in the usual sense, since we randomly sample
    the data set (vs methodically marching through it)                
    """
    sz = model.input_shape[-2:]
    score_all = []

    for ii in range(n_epochs):
        print('[train_model]: starting "epoch" %d (of %d)' % (ii, n_epochs))

        for jj in print_generator(range(n_mb_per_epoch)):
            Xi, Yi = random_minibatch(X_train, Y_train, mb_size, sz, xform)
            Yi = pixelwise_one_hot(Yi) 
            loss, f1 = model.train_on_batch(Xi, Yi)
            score_all.append(f1)

        # save state
        fn_out = 'weights_epoch%04d.hdf5' % ii
        model.save_weights(fn_out)

        # evaluate performance on validation data
        Yi_hat = deploy_model(X_valid, model)
        np.savez('valid_epoch%04d' % ii, X=X_valid, Y=Y_valid, Y_hat=Yi_hat, s=score_all)

        if len(np.unique(Y_valid.flatten())) == 2:
            pred = (Yi_hat[:,1,:,:].flatten() >= 0.5).astype(np.int32)
            print('[train_model]: f1 on validation data:    %0.3f' % skm.f1_score(Y_valid.flatten(), pred))
        print('[train_model]: recent train performance: %0.3f' % np.mean(score_all[-20:]))
        print('[train_model]: y_hat min, max, frac_0:   %0.2f / %0.2f / %0.3f' % (np.min(Yi_hat), np.max(Yi_hat), 1. * np.sum(Yi_hat[:,0,...]) / Yi_hat[:,0,...].size))

    return score_all



def deploy_model(X, model):
    """
    X : a tensor of dimensions (n_examples, n_channels, n_rows, n_cols)

    Note: n_examples will be used as the minibatch size.

    Note: we could be more sophisticated here and, instead of partitioning X,
          calculate with some overlaps and average to mitigate edge effects.
    """
    # the only slight complication is that the spatial dimensions of X might
    # not be a multiple of the tile size.
    sz = model.input_shape[-2:]

    Y_hat = None  # delay initialization until we know the # of classes

    for rr in range(0, X.shape[-2], sz[0]):
        ra = rr if rr+sz[0] < X.shape[-2] else X.shape[-2] - sz[0]
        rb = ra+sz[0]
        for cc in range(0, X.shape[-1], sz[1]):
            ca = cc if cc+sz[1] < X.shape[-1] else X.shape[-1] - sz[-1]
            cb = ca+sz[1]
            Y_hat_mb = model.predict(X[:, :, ra:rb, ca:cb])

            # create Y_hat if needed
            if Y_hat is None:
                Y_hat = np.zeros((X.shape[0], Y_hat_mb.shape[1], X.shape[2], X.shape[3]), dtype=Y_hat_mb.dtype)

            # store this mini-batch
            Y_hat[:,:,ra:rb,ca:cb] = Y_hat_mb

    return Y_hat    
