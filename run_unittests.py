

__author__ = 'mjp, Oct 2016'
__license__ = 'Apache 2.0'


import unittest
import numpy as np

from data_tools import *
from train import f1_score



class TestStuff(unittest.TestCase):
    def test_f1_score(self):
        y_true = np.ones((100,))
        y_hat = np.ones((100,))
        self.assertTrue(np.abs(f1_score(y_true, y_hat) - 1.0) < 1e-8)
        self.assertTrue(np.abs(f1_score(y_true, 0*y_hat) - 0.0) < 1e-8)

        y_hat[0:50] = 0
        self.assertTrue(np.abs(f1_score(y_true, y_hat) - 2./3) < 1e-8)

        


if __name__ == '__main__':
    unittest.main()
