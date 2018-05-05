#!/usr/bin/env python
"""
File: image_types
Date: 5/4/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

from BraTS.load_utils import *

from enum import Enum

image_shape = (240, 240, 155)
mri_shape = (len(image_shape),) + image_shape


class ImageType(Enum):
    t1 = 1
    t2 = 2
    t1ce = 3
    flair = 4
    seg = 5


image_types = [ImageType.t1, ImageType.t2, ImageType.t1ce,
               ImageType.flair, ImageType.seg]

image_type_names = {ImageType.t1: "t1",
                    ImageType.t2: "t2",
                    ImageType.t1ce: "t1ce",
                    ImageType.flair: "flair",
                    ImageType.seg: "seg"}

mri_indices = {ImageType.t1: 0,
               ImageType.t2: 1,
               ImageType.t1ce: 2,
               ImageType.flair: 3}


def get_image_type(image_file):
    for img_type, name in image_type_names.items():
        if "%s." % name in image_file:
            return img_type
    return None


def get_image_file_map(patient_dir):
    files = listdir(patient_dir)
    d = {}
    for file in files:
        d[get_image_type(file)] = file
    return d
