#!/usr/bin/env python
"""
File: Patient
Date: 5/1/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

from BraTS.image_types import *

img_shape = (240, 240, 155)
mri_shape = (len(image_types),) + img_shape

import numpy as np
import nibabel as nib

from BraTS.load_utils import *


def load_patient_data(patient_data_dir,
                      mri_array=None, seg_array=None, index=None):
    """
    Load a single patient's image data

    :param patient_data_dir: Directory containing image data
    :return: Tuple containing a tf.Tensor containing MRI data
    """

    load_inplace = mri_array is not None and seg_array is not None

    if not load_inplace:
        mri_data = np.empty(shape=mri_shape)

    for img_file in listdir(patient_data_dir):
        img = nib.load(img_file).get_data()
        img_type = get_image_type(img_file)

        if img_type == ImageType.seg:
            seg_data = img
            continue

        channel_index = mri_indices[img_type]

        if load_inplace:
            mri_array[index, channel_index] = img
        else:
            mri_data[channel_index] = img

    # Load segmentation data
    if load_inplace:
        seg_array[index] = seg_data
    else:
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
