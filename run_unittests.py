

__author__ = 'mjp, Oct 2016'
__license__ = 'Apache 2.0'


import unittest
import numpy as np

from data_tools import *
from cnn_tools import f1_score



class TestStuff(unittest.TestCase):
    def test_f1_score(self):
        y_true = np.ones((100,))
        y_hat = np.ones((100,))
        self.assertTrue(np.abs(f1_score(y_true, y_hat) - 1.0) < 1e-8)

        y_hat[0:50] = 0
        self.assertTrue(np.abs(f1_score(y_true, y_hat) - 2./3) < 1e-8)


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
                    
        


if __name__ == '__main__':
    unittest.main()
