#!/usr/bin/env python
"""
File: BraTS_partitions
Date: 5/5/18 
Author: Jon Deaton (jdeaton@stanford.edu)

This file exports the training, testing and validation data-set partition
for training and evaluating the BraTS segmentation model.

Import this script to accomplish any of the following
    - Generate a new set of IDs designated to each category
    -
    -
"""

import os
import BraTS
from BraTS.modalities import mri_shape, image_shape

from random import shuffle

train_ids_filename = "train_ids"
test_ids_filename = "test_ids"
validation_ids_filename = "validation_ids"

default_partition_store = os.path.join(os.path.split(__file__)[0], "BraTS_partition")
default_brats_year = 2018


def generate_random_partitioning(brats_root, output_dir, year, num_test=40, num_validation=40):
    """
    Generates and saves a new partitioning of the specified BraTS data-set into training, test
    and validation sets and saves them to file

    :param brats_root: The root directory of the BraTS data set
    :param output_dir: Output directory to save the partition ids
    :param year: The BraTS year to use
    :param num_test: The number of test examples
    :param num_validation: The number of validation examples
    :return: None
    """

    # Load up the brats data set
    brats = BraTS.DataSet(brats_root=brats_root, year=year)

    # Shuffle up the ids for a homogeneous split
    ids = brats.train.ids
    shuffle(ids)

    # Split up the patient IDs into test, validation and train
    test_ids = ids[:num_test]
    validation_ids = ids[num_test:(num_test + num_validation)]
    train_ids = set(ids) - set(test_ids) - set(validation_ids)

    train_ids_file = os.path.join(output_dir, train_ids_filename)
    test_ids_file = os.path.join(output_dir, test_ids_filename)
    validation_ids_file = os.path.join(output_dir, validation_ids_filename)

    # Write the test and validation IDs to file
    with open(train_ids_file, 'w') as f:
        f.write("\n".join(train_ids))

    with open(test_ids_file, 'w') as f:
        f.write("\n".join(test_ids))

    with open(validation_ids_file, 'w') as f:
        f.write("\n".join(validation_ids))


# Functions for retrieving the partition, once generated
def get_all_partition_ids(partition_dir=default_partition_store):
    train_ids = get_training_ids(partition_dir)
    test_ids = get_test_ids(partition_dir)
    validation_ids = get_validation_ids(partition_dir)
    return train_ids, test_ids, validation_ids

def get_training_ids(partition_dir=default_partition_store):
    train_ids_file = os.path.join(partition_dir, train_ids_filename)
    return _get_ids(train_ids_file)


def get_test_ids(partition_dir=default_partition_store):
    test_ids_file = os.path.join(partition_dir, test_ids_filename)
    return _get_ids(test_ids_file)


def get_validation_ids(partition_dir=default_partition_store):
    validation_ids_file = os.path.join(partition_dir, validation_ids_filename)
    return _get_ids(validation_ids_file)


def _get_ids(ids_file):
    """
    Returns a list of IDs found in an ID file
    :param ids_file: Path to a file containing tab/space/line seperated IDs
    :return: A set of IDs
    """
    assert isinstance(ids_file, str)
    if not os.path.exists(ids_file):
        raise FileNotFoundError(ids_file)
    with open(ids_file, 'r') as f:
        return set(f.read().split())
