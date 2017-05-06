

__author__ = 'mjp, Oct 2016'
__license__ = 'Apache 2.0'


import unittest
import numpy as np

from data_tools import pixelwise_one_hot
from cnn_tools import f1_score, pixelwise_ace_loss

import theano



class TestStuff(unittest.TestCase):
    
    def test_f1_score(self):
        n_examples = 100
        n_classes = 2
        m,n = 16, 16    # num pixels
        n2 = np.floor(n/2).astype(np.int32)
        n4 = np.floor(n/4).astype(np.int32)

        y_true = np.zeros((n_examples, 1, m, n))
        y_true[:,:,:,0:n2] = 1
        y_true = pixelwise_one_hot(y_true).astype(np.float32)

        # Note: we assume a theano backend here
        a = theano.tensor.tensor4('y_true')
        b = theano.tensor.tensor4('y_hat')
        f_theano = f1_score(a,b)
        
        result = f_theano.eval({a : y_true, b : y_true})
        self.assertTrue(np.abs(result - 1.0) < 1e-12)

        # case where all estimates are incorrect
        y_totally_wrong = 1 - y_true
        result = f_theano.eval({a : y_true, b : 1.0 - y_true})
        self.assertTrue(np.abs(result - 0.0) < 1e-12)

        # kill recall performance by 1/2
        y_recall_half = np.copy(y_true)
        y_recall_half[:,:,:,0:n4] =  1 - y_recall_half[:,:,:,0:n4]
        expected = 2./3      # 2 * (.5 * 1) / (.5 + 1)
        result = f_theano.eval({a : y_true, b : y_recall_half})
        self.assertTrue(np.abs(result - expected) < 1e-5)


        
    def test_y_onehot(self):
        y_fake = np.random.randint(low=0, high=10, size=(5,1,3,3))
        y_onehot = pixelwise_one_hot(y_fake)
 
        self.assertTrue(y_onehot.shape[1] == 10)

        for kk in range(y_fake.shape[0]):
            for ii in range(y_fake.shape[2]):
                for jj in range(y_fake.shape[3]):
                    yi = y_fake[kk,0,ii,jj]
                    self.assertTrue(y_onehot[kk,yi,ii,jj])
                    self.assertTrue(np.sum(y_onehot[kk,:,ii,jj]) == 1)

        # make sure missing labels work as expected
        y_fake = np.random.randint(low=0, high=10, size=(5,1,3,3))
        y_fake[0,:,:,:] = -10
        y_onehot = pixelwise_one_hot(y_fake)
        assert(np.all(y_onehot[0,:,:,:] == 0))

        
    def test_pixelwise_ace_loss(self):
        n_examples = 100
        n_classes = 5
        m,n = 16, 16    # num pixels
        n2 = np.floor(n/2).astype(np.int32)
        n4 = np.floor(n/4).astype(np.int32)

        y_true_raw = np.random.randint(low=0, high=n_classes, size=(n_examples, 1, m, n))
        y_true = pixelwise_one_hot(y_true_raw).astype(np.float32)

        # Note: we assume theano backend here
        a = theano.tensor.tensor4('y_true')
        b = theano.tensor.tensor4('y_hat')
        f_theano = pixelwise_ace_loss(a,b)

        #--------------------------------------------------
        # perfect estimate 
        #--------------------------------------------------
        lossPerfect = f_theano.eval({a : y_true, b : y_true})
        self.assertTrue(lossPerfect > 0)

        #--------------------------------------------------
        # If there are no class labels whatsoever, 1/N_labeled = inf
        #--------------------------------------------------
        lossNL = f_theano.eval({a : np.zeros(y_true.shape, dtype=np.float32), b : y_true})
        print(lossNL)
        self.assertTrue(np.isinf(lossNL))

        #--------------------------------------------------
        # make some mistakes and see the loss is higher
        #--------------------------------------------------
        y_hat = np.copy(y_true)
        splatter = np.random.randint(low=0, high=y_hat.size, size=(500,))
        tmp = y_hat.ravel()
        tmp[splatter] = 1
        y_hat = np.reshape(tmp, y_hat.shape)
        lossErr = f_theano.eval({a : y_true, b : y_hat})
        self.assertTrue(lossPerfect < lossErr)

        #--------------------------------------------------
        # here we test to see that an error is worse from
        # a loss perspective than an unlabeled pixel.
        #--------------------------------------------------
        supp = np.random.randint(low=0, high=y_true_raw.size, size=(500,))
        y_true_u = np.copy(y_true_raw).flatten()
        y_true_u[supp] = -1

        y_hat = np.copy(y_true_u).flatten()
        y_hat[supp] = np.mod(y_true_u[supp] + 1, n_classes)  # all in error
        
        y_true_u = np.reshape(y_true_u, y_true_raw.shape)
        y_hat = np.reshape(y_hat, y_true_u.shape)
        
        y_true_u = pixelwise_one_hot(y_true_u).astype(np.float32)
        y_hat = pixelwise_one_hot(y_hat).astype(np.float32)
        
        loss1 = f_theano.eval({a : y_true, b : y_hat})
        loss2 = f_theano.eval({a : y_true_u, b : y_hat})
        self.assertTrue(loss2 < loss1)
        






if __name__ == '__main__':
    unittest.main()
