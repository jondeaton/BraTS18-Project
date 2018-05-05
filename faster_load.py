#!/usr/bin/env python
"""
Trying to see if I can get it to load faster...
"""


import os
import numpy as np
import nibabel as nib
import timeit

import cProfile, pstats, io

from BraTS.image_types import *
from BraTS.Patient import *

dir = '/Users/jonpdeaton/Datasets/BraTS/BraTS17/BraTS17_Training/HGG'

cache = {}


def load_fast(patient_dir):
    for file in listdir(patient_dir):
        cache[file] = nib.load(file).get_data()

n = 10
def test():
    i = 0
    for i, patient_dir in enumerate(listdir(dir)):
        if i == n:
            break

        print("Loading: %s" % os.path.split(patient_dir)[1])

        mri_array = np.empty(shape=(n,) + mri_shape)
        seg_array = np.empty(shape=(n,) + img_shape)

        # load_fast(patient_dir)
        cache[patient_dir] = load_patient_data(patient_dir,
                                               mri_array=mri_array,
                                               seg_array=seg_array,
                                               index=i)



pr = cProfile.Profile()
pr.enable()

s = timeit.timeit(test, number=1)
print("time: %s sec" % s)

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())


