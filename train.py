"""
   Trains a dense (per-pixel) classifier.
"""

from __future__ import print_function

__author__ = 'mjp, Oct 2016'
__license__ = 'Apache 2.0'



import os, sys, time

import numpy as np
from scipy.interpolate import interp2d

np.random.seed(9999)

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

from data_tools import *




def f1_score(y_true, y_hat):
    """ Note: this works for keras objects (e.g. during training) or 
              on numpy objects.
    """
    # TODO: discrete vs continuous y_hat values
    try: 
        # default is to assume a Keras object
        y_true_flat = K.flatten(y_true)
        y_hat_flat = K.flatten(y_hat)
    
        intersection = K.sum(y_hat_flat * y_true_flat) 
        precision = intersection / K.sum(y_hat_flat)
        recall = intersection / K.sum(y_true_flat)
    except AttributeError:
        # probably was a numpy array instead
        y_true_flat = y_true.flatten()
        y_hat_flat = y_hat.flatten()
    
        intersection = np.sum(y_hat_flat * y_true_flat) 
        precision = intersection / np.sum(y_hat_flat)
        recall = intersection / np.sum(y_true_flat)
        
    f1 = 2 * precision * recall / (precision + recall) 
    return f1



def f1_score_loss(y_true, y_hat):
    return -f1_score(y_true, y_hat)



def create_unet(sz):
    """
      sz : a tuple specifying the input image size in the form:
           (# channels, # rows, # columns)
      
      References:  
        1. Ronneberger et al. "U-Net: Convolutional Networks for Biomedical
           Image Segmentation." 2015. 
        2. https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
    """
    assert(len(sz) == 3)

    bm = 'same'
    
    inputs = Input(sz)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode=bm)(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode=bm)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode=bm)(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode=bm)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode=bm)(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode=bm)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode=bm)(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode=bm)(conv4)
    conv4 = Dropout(.5)(conv4) # mjp
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode=bm)(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode=bm)(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode=bm)(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode=bm)(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode=bm)(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode=bm)(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode=bm)(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode=bm)(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode=bm)(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode=bm)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=f1_score_loss, metrics=[f1_score])

    return model



def train_model(X_train, Y_train, X_valid, Y_valid, model,
                n_epochs=30, n_mb_per_epoch=100, sz=(256, 256)):
    """
    Note: these are not epochs in the usual sense, since we randomly sample
    the data set (vs methodically marching through it)                
    """
    acc_all = []

    for ii in range(n_epochs):
        print('starting "epoch" %d (of %d)' % (ii, n_epochs))

        for jj in timed_collection(range(n_mb_per_epoch)):
            Xi, Yi = random_minibatch(X_train, Y_train, 30, sz=sz)
            loss, acc = model.train_on_batch(Xi, Yi)
            acc_all.append(acc)

        # save state
        fn_out = 'weights_epoch%04d.hdf5' % ii
        model.save_weights(fn_out)

        # evaluate performance on (a subset of) validation data
        Xi, Yi = random_crop([X_valid, Y_valid], sz)
        Yi_hat = model.predict(Xi)
        print('f1 on validation data: %0.3f' % f1_score(Yi, Yi_hat))
        np.save('valid_epoch%04d.npy' % ii, Xi, Yi, Yi_hat)

    return acc_all



def timed_collection(c, rate=60*2):
    """ Provides status on progress as one iterates through a collection.
    """
    start_time = time.time()
    last_chatter = -rate

    for idx, ci in enumerate(c):
        yield ci
        
        elapsed = time.time() - start_time
        if (elapsed) > last_chatter + rate:
            last_chatter = elapsed
            print('finished %d items in %0.2f minutes' % (idx, elapsed/60.))
                 
    
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    tile_size = (256, 256)
    
    # load raw data
    isbi_dir = os.path.expanduser('~/Data/ISBI-2012')
    X_train = load_multilayer_tiff(os.path.join(isbi_dir, 'train-volume.tif'))
    Y_train = load_multilayer_tiff(os.path.join(isbi_dir, 'train-labels.tif'))
    
    Y_train = 1 - Y_train / 255.  # map to [0 1] and make 1 \equiv membrane

    # TODO: normalize image data??

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
    model = create_unet((1, tile_size[0], tile_size[1]))
    train_model(X_train, Y_train, X_valid, Y_valid, model, n_epochs=30, sz=tile_size)

