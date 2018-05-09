#!/usr/bin/env python
"""
File: partitions
Date: 5/8/18 
Author: Jon Deaton (jdeaton@stanford.edu)

-----------------------------------------

Yields the training, testing, and validation TFRecordDatasets

"""

import os
import tensorflow as tf
from .partitioning import default_partition_store, get_training_ids, get_test_ids, get_validation_ids

from preprocessing.records import *


def get_record_id_map(brats_TFRecords_dir):
    """
    Get the mapping from patient_id --> TFRecord file

    :param brats_TFRecords_dir: directory containing all TFRecords
    :return: Dictionary mapping patient_id to TFRecord file
    """
    tfrecord_filenames = os.listdir(brats_TFRecords_dir)
    id_record_map = {}
    for file_name in tfrecord_filenames:
        patient_id = get_id_of_TFRecord(file_name)
        id_record_map[patient_id] = os.path.join(brats_TFRecords_dir, file_name)
    return id_record_map


def load_datasets(brats_TFRecords_dir, partition_dir=default_partition_store):
    """
    Loads the training, test, and validation data-sets into TensorFlow TFRecordDataset

    :param partition_dir: directory of partition
    :param brats_TFRecords_dir: Directory of TFRecords
    :return: TFRecordDatasets:
        - train_dataset
        - test_dataset
        - validation_dataset
    """
    train_ids = get_training_ids(partition_dir)
    test_ids = get_test_ids(partition_dir)
    validation_ids = get_validation_ids(partition_dir)

    record_map = get_record_id_map(brats_TFRecords_dir)
    train_dataset = get_dataset(train_ids, record_map)
    test_dataset = get_dataset(test_ids, record_map)
    validation_dataset = get_dataset(validation_ids, record_map)

    return train_dataset, test_dataset, validation_dataset


def get_dataset(patient_ids, record_map):
    """
    Makes a single TFRecordDataset containing the specified IDs

    :param patient_ids: The patient IDs to put in
    :param record_map: Mapping from patient_id to TFRecord filename
    :return: TFRecordDataset containing the TFRecords specified
    """
    filenames = []
    for patient_id in patient_ids:
        filenames.append(record_map[patient_id])
    return tf.data.TFRecordDataset(filenames=filenames)

