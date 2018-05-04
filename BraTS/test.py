#!/usr/bin/env python
"""
File: test
Date: 5/3/18 
Author: Jon Deaton (jdeaton@stanford.edu)

This file tests the functionality of the BraTS dataset loader
"""

import BraTS
import unittest
import numpy as np

# If, for some reason, you wanted to test this on your machine you would
# need to set up the BraTS data-sets in some directory and set that path here
brats_root = "/Users/jonpdeaton/Datasets/BraTS"

BraTS.set_root(brats_root)


class BraTSTest(unittest.TestCase):

    def test_train_images(self):
        brats = BraTS.DataSet(year=2017)
        train_mris = brats.train.mris
        self.assertIsInstance(train_mris, np.ndarray)


if __name__ == "__main__":
    unittest.main()