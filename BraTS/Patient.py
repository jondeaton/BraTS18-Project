#!/usr/bin/env python
"""
File: Patient
Date: 5/1/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

mri_shape = (4,) + img_shape

import numpy as np
import nibabel as nib
from nibabel.testing import data_path

from BraTS.load_utils import *


def load_patient_data(patient_data_dir):
    """

    :param patient_data_dir:
    :return: Tuple containing a tf.Tensor containing MRI data
    """

    # Load Flair, T1, T1-ce, T2 into a Numpy Array
    mri_data = np.empty(shape=mri_shape)
    for i, keyword in enumerate(("flair", "t1.", "t1ce.", "t2")):
        img_file = find_file_containing(patient_data_dir, keyword)
        if img_file is None:
            raise Exception("Could not find %s image file for patient %s" % (keyword, patient_data_dir))

        img = nib.load(img_file).get_data()
        if img.shape != mri_shape:
            raise Exception("Unexpected image shape %s in file %s" % (img.shape, img_file))
        mri_data[i] = img

    # Load segmentation data
    seg_file = find_file_containing(patient_data_dir, "seg")
    if seg_file is None:
        raise Exception("Couldn't find segmentation data for patient %s" % patient_data_dir)
    seg_data = nib.load(seg_file).get_data()
    return mri_data, seg_data


class Patient:
    def __init__(self, id, age=None, survival=None, mri=None, seg=None):
        self.id = id
        self.age = age
        self.survival = survival
        self.mri = mri
        self.seg = seg

    @property
    def flair(self):
        if not isinstance(self.mri, np.ndarray):
            raise Exception("patient %s MRI is not a numpy array." % self.id)
        return self.mri[0]

    @property
    def t1(self):
        if not isinstance(self.mri, np.ndarray):
            raise Exception("patient %s MRI is not a numpy array." % self.id)
        return self.mri[1]

    @property
    def t1ce(self):
        if not isinstance(self.mri, np.ndarray):
            raise Exception("patient %s MRI is not a numpy array." % self.id)
        return self.mri[2]

    @property
    def t2(self):
        if not isinstance(self.mri, np.ndarray):
            raise Exception("patient %s MRI is not a numpy array." % self.id)
        return self.mri[3]
