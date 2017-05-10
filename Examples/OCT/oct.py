"""
Some functions for working with OCT data.

REFERENCES:
   [Shen] "Tools for NIfTI and ANALYZE image", https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
"""

from __future__ import print_function

__author__ = 'mjp, March 2017'
__license__ = 'Apache 2.0'

import numpy as np
import h5py



def load_oct_sample_data(h5_fn):
    """ I exported this data manually from the .nii file format using Matlab 
    and the third party library [Shen].
    """
    h5 = h5py.File(h5_fn, 'r')

    # this file contains just two annotated slices
    x_60 = h5['x_60'].value.T
    y_60 = h5['y_60'].value.T
    x_70 = h5['x_70'].value.T
    y_70 = h5['y_70'].value.T

    # pack into tensors
    def to_tensor(*args):
        X = np.zeros((len(args), args[0].shape[0], args[0].shape[1]))
        for ii in range(len(args)):
            X[ii,...] = args[ii]
        return X

    X = to_tensor(x_60, x_70)
    Y = to_tensor(y_60, y_70)

    # normalize
    X = X / 255.

    # add a color channel dimension
    X = X[:, np.newaxis, :, :]

    return X, Y.astype(np.uint32)
