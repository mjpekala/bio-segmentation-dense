"""
   Trains a dense (per-pixel) classifier.
"""


from __future__ import print_function

__author__ = 'mjp, Oct 2016'
__license__ = 'Apache 2.0'



import os, sys
import pdb
from PIL import Image
import numpy as np

np.random.seed(9999)

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K


#-------------------------------------------------------------------------------
# Utility/helper functions
#-------------------------------------------------------------------------------

def load_multilayer_tiff(data_file):
    """ Loads data from a grayscale multilayer .tif file.

    data_file    := the tiff file name
    add_feat_dim := adds a new "feature dimensinon" to grayscale data

    RETURNS:
       A numpy tensor with dimensions:
           (#images, width, height)
       -or-
           (#images, 1, width, height)
    """
    if not os.path.isfile(data_file):
        raise RuntimeError('could not find file "%s"' % data_file)
    
    # load the data from multi-layer TIF files
    img = Image.open(data_file)
    X = [];
    for ii in xrange(sys.maxint):
        Xi = np.array(img, dtype=np.float32)
        if Xi.ndim == 2:
            Xi = Xi[np.newaxis, ...] # add slice dimension
        X.append(Xi)
        try:
            img.seek(img.tell()+1)
        except EOFError:
            break # this just means hit end of file (not really an error)

    # list of images -> 3d tensor
    X = np.concatenate(X, axis=0) 

    # add a channel dimension 
    if X.ndim == 3:
        X = X[:, np.newaxis, :, :]
        
    return X


def minibatch_indices(n_examples, mb_size=10):
    idx = np.arange(n_examples)
    np.random.shuffle(idx)

    for ii in range(0, n_examples, mb_size):
        n_this_batch = min(mb_size, n_examples-ii)
        yield idx[ii:ii+n_this_batch]


def apply_symmetry(X):
    """Implements synthetic data augmentation by randomly appling
    an element of the group of symmetries of the square to a single 
    mini-batch of data.

    The default set of data augmentation operations correspond to
    the symmetries of the square (a non abelian group).  The
    elements of this group are:

      o four rotations (0, pi/2, pi, 3*pi/4)
        Denote these by: R0 R1 R2 R3

      o two mirror images (about y-axis or x-axis)
        Denote these by: M1 M2

      o two diagonal flips (about y=-x or y=x)
        Denote these by: D1 D2

    This page has a nice visual depiction:
      http://www.cs.umb.edu/~eb/d4/


    Parameters: 
       X := Mini-batch data (#examples, #channels, rows, colums) 
    """

    def R0(X):
        return X  # this is the identity map

    def M1(X):
        return X[:,:,::-1,:]

    def M2(X): 
        return X[:,:,:,::-1]

    def D1(X):
        return np.transpose(X, [0, 1, 3, 2])

    def R1(X):
        return D1(M2(X))   # = rot90 on the last two dimensions

    def R2(X):
        return M2(M1(X))

    def R3(X): 
        return D2(M2(X))

    def D2(X):
        return R1(M1(X))


    symmetries = [R0, R1, R2, R3, M1, M2, D1, D2]
    op = np.random.choice(symmetries) 
        
    # For some reason, the implementation of row and column reversals, 
    #     e.g.      X[:,:,::-1,:]
    # break PyCaffe.  Numpy must be doing something under the hood 
    # (e.g. changing from C order to Fortran order) to implement this 
    # efficiently which is incompatible w/ PyCaffe.  
    # Hence the explicit construction of X2 with order 'C' below.
    #
    # Not sure this matters for Theano/Keras, but leave in place anyway.
    X2 = np.zeros(X.shape, dtype=np.float32, order='C') 
    X2[...] = op(X)

    return X2



#-------------------------------------------------------------------------------
# CNN functions
#-------------------------------------------------------------------------------

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
    
    inputs = Input(sz)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=f1_score_loss, metrics=[f1_score])

    return model



def train_model(X_train, Y_train, X_valid, Y_valid, model, n_epochs=30):
    acc_all = []
    
    for ii in range(n_epochs):
        print('starting epoch %d (of %d)' % (ii, n_epochs))

        for idx in minibatch_indices(X_train.shape[0]):
            Xi = X_train[idx,...]
            Yi = Y_train[idx,...]

            # data augmentation 
            Xi = apply_symmetry(Xi)

            # TODO: apply elastic deformations to X and Y

            # train this mini-batch
            loss, acc = model.train_on_batch(Xi, Yi)
            acc_all.append(acc)

        # save state
        fn_out = 'weights_%04d.hdf5' % ii
        model.save_weights(fn_out)

        # evaluate performance on validation data
        Y_hat_valid = model.predict(X_valid)
        print('f1 on validation data: %0.3f' % f1_score(Y_valid, Y_hat_valid))
        np.save('y_valid_hat_%04d.npy' % ii, Y_hat_valid)

    return acc_all

             
    
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    
    # load raw data
    isbi_dir = os.path.expanduser('~/Data/ISBI-2012')
    X_train = load_multilayer_tiff(os.path.join(isbi_dir, 'train-volume.tif'))
    Y_train = load_multilayer_tiff(os.path.join(isbi_dir, 'train-labels.tif'))
    
    Y_train = 1 - Y_train / 255.  # map to [0 1] and make 1 \equiv membrane

    # split into train and valid
    train_slices = range(20)
    valid_slices = range(20,30)
    X_valid = X_train[valid_slices,:,:,:]
    Y_valid = Y_train[valid_slices,:,:]
    X_train = X_train[train_slices,:,:,:]
    Y_train = Y_train[train_slices,:,:]

    # train model 
    acc = model = create_unet((1, X_train.shape[-2], X_train.shape[-1]))
    train_model(X_train, Y_train, X_valid, Y_valid, model, n_epochs=300)

