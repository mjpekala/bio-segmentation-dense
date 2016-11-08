"""
Some functions for manipulating data/images.
"""

from __future__ import print_function

__author__ = 'mjp, Oct 2016'
__license__ = 'Apache 2.0'


import os, sys
import numpy as np

from PIL import Image
import pylab as plt
from scipy.interpolate import griddata




def load_multilayer_tiff(data_file):
    """Loads data from a grayscale multilayer .tif file.

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



def apply_symmetries(X):
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


def make_displacement_mesh(n, sigma=20, n_seed_points=5):
    """ Creates a warping/displacement mesh (for synthetic data augmentation).
    
    Parameters:
      n     : The width/height of the target image (assumed to be square)
      sigma : standard deviation of displacements. 
              If negative, is interpreted as a deterministic displacement.
              This latter usage is for testing, not actual applications.
      n_seed_points : The number of random displacements to choose (in each dimension).
              Displacements at all locations will be obtained via interpolation.
    """
    glue = lambda X, Y: np.vstack([X.flatten(), Y.flatten()]).transpose()

    # the domain Omega is [0:n)^2
    omega_x, omega_y = np.meshgrid(np.arange(n), np.arange(n))

    # create random displacement in the domain.
    d_pts = np.linspace(0, n, n_seed_points)
    d_xx, d_yy = np.meshgrid(d_pts, d_pts)

    if sigma > 0:
        # random displacement
        dx = sigma * np.random.randn(d_xx.size)
        dy = sigma * np.random.randn(d_xx.size)
    else:
        # deterministic displacement (for testing)
        dx = abs(sigma) * np.ones(d_xx.size)
        dy = abs(sigma) * np.ones(d_yy.size)
    
    # use interpolation to generate a smooth displacement field.
    omega_dx = griddata(glue(d_xx, d_yy), dx.flatten(), glue(omega_x, omega_y))
    omega_dy = griddata(glue(d_xx, d_yy), dy.flatten(), glue(omega_x, omega_y))

    # reshape 1d -> 2d
    omega_dx = np.reshape(omega_dx, (n,n))
    omega_dy = np.reshape(omega_dy, (n,n))

    # generate a perturbed mesh
    omega_xnew = omega_x + omega_dx
    omega_ynew = omega_y + omega_dy

    return omega_xnew, omega_ynew



def plot_mesh(xx, yy, linespec='k-'):
    assert(xx.ndim == 2);  assert(yy.ndim == 2)
    plt.hold(True)
    for r in range(xx.shape[0]):
        for c in range(xx.shape[1]):
            if c+1 < xx.shape[1]: plt.plot((xx[r,c], xx[r,c+1]), (yy[r,c], yy[r,c]), 'k-') # east
            if r+1 < xx.shape[0]: plt.plot((xx[r,c], xx[r,c]), (yy[r,c], yy[r+1,c]), 'k-') # south
    plt.gca().set_xlim([np.min(xx), np.max(xx)])
    plt.gca().set_ylim([np.min(yy), np.max(yy)])
    plt.hold(False)

    

def apply_displacement_mesh(X, omega_xnew, omega_ynew):
    glue = lambda X, Y: np.vstack([X.flatten(), Y.flatten()]).transpose()
    n = X.shape[0]
    assert(X.ndim == 2 and n == X.shape[1])
    
    omega_x, omega_y = np.meshgrid(np.arange(n), np.arange(n))
    
    # use interpolation to estimate pixel intensities on original lattice
    X_new = griddata(glue(omega_xnew, omega_ynew),
                     X.flatten(),
                     glue(omega_x, omega_y))
    X_new = np.reshape(X_new, (n,n))

    return X_new
