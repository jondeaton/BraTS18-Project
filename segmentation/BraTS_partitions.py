#!/usr/bin/env python
"""
File: BraTS_partitions
Date: 5/5/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import tensorflow as tf

import BraTS
from BraTS.image_types import mri_shape, image_shape

partition_store = os.path.join(os.path.split(__file__)[0], "BraTS_partitions")
test_ids_filename = "test_ids"
validation_ids_filename = "validation_ids"
brats_year = 2018


def set_partition_store(new_store):
    assert isinstance(new_store, str)
    global partition_store
    partition_store = new_store


def train_dataset_gen():
    test_ids = get_ids(test_ids_filename)
    validation_ids = get_ids(validation_ids_filename)
    brats = BraTS.DataSet(year=brats_year)

    train_ids = set(brats.train.ids) - test_ids - validation_ids
    for id in train_ids:
        patient = brats.train.patient(id)
        yield (patient.mri, patient.seg)


def test_dataset_gen():
    test_ids = get_ids(test_ids_filename)
    brats = BraTS.DataSet(year=brats_year)
    for id in test_ids:
        patient = brats.train.patient(id)
        yield (patient.mri, patient.seg)


def get_test_Dataset():
    return tf.data.Dataset().from_generator(train_dataset_gen,
                                            output_types=(tf.float32,tf.int32),
                                            output_shapes=(mri_shape, image_shape))

def get_ids(ids_filename):
    full_path = os.path.join(partition_store, ids_filename)
    with open(full_path, 'r') as f:
        return set(f.readlines())
