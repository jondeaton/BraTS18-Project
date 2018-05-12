#!/usr/bin/env python
"""
File: partitions
Date: 5/8/18 
Author: Jon Deaton (jdeaton@stanford.edu)

-----------------------------------------

Yields the training, testing, and validation tf.data.Dataset

"""

import os
import tensorflow as tf
from .partitioning import default_partition_store, get_training_ids, get_test_ids, get_validation_ids
from preprocessing.records import get_id_of_TFRecord

from BraTS.Patient import load_patient_data
from BraTS.modalities import get_modality_file, Modality

def load_patient_dir_wrapper(patient_dir, seg_file):
    mri, seg = load_patient_data()

    return

def load_datasets(directory_map, partition_dir=default_partition_store):
    """
    Loads the training, test, and validation data-sets into TensorFlow TFRecordDataset

    :param partition_dir: Directory containing files with ids for each partition
    :return: train_dataset, test_dataset, validation_dataset
    """

    # Get ethe designated training IDs for each partition
    train_ids = get_training_ids(partition_dir)
    test_ids = get_test_ids(partition_dir)
    validation_ids = get_validation_ids(partition_dir)

    train_dataset = get_dataset(train_ids, directory_map)
    test_dataset = get_dataset(test_ids, directory_map)
    validation_dataset = get_dataset(validation_ids, directory_map)
    return train_dataset, test_dataset, validation_dataset

def get_dataset(patient_ids, directory_map):
    """
    Creates a TensorFlow DataSet

    :param patient_ids:
    :param directory_map:
    :return:
    """

    # patient_dirs = tf.placeholder(tf.string, shape=[None])
    # seg_files = tf.placeholder(tf.string, shape=[None])

    patient_dirs = [directory_map[patient_id] for patient_id in patient_ids]
    segs = [get_modality_file(dir, Modality.seg) for dir in patient_dirs]

    dataset = tf.data.Dataset.from_tensor_slices((patient_dirs, segs))

    py_func = lambda patient_dir, seg_file: \
        tf.py_func(load_patient_dir_wrapper,
                   [patient_dir, seg_file],
                   [tf.Tensor, tf.Tensor])

    mapped = dataset.map(py_func)
    return mapped


# The following functions are used for loading the TFRecordDataset
# I don't think that we should use this methodology anymore though so


def _parse_function(serialized):
    mri = tf.train.Feature(float_list=tf.train.FloatList())
    seg = tf.train.Feature(int64_list=tf.train.Int64List())

    features = {'train/mri': mri, 'train/seg': seg}

    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    mri_raw = parsed_example['train/mri']
    mri_T  = tf.decode_raw(mri_raw, tf.uint8)
    mri_T = tf.cast(mri_T, tf.float32)

    seg_raw = parsed_example['train/seg']
    seg_T = tf.decode_raw(seg_raw, tf.uint8)
    seg_T = tf.cast(seg_T, tf.int64)

    return mri_T, seg_T


from BraTS.modalities import mri_shape, seg_shape


def _reshape_fn(mri, seg):
    _mri = tf.reshape(mri,shape=(None) + mri_shape)
    _seg = tf.reshape(seg, shape=(None) + seg_shape)
    return _mri, _seg



def load_tfrecord_datasets(brats_tfrecords_dir, partition_dir=default_partition_store):
    """
    Loads the training, test, and validation data-sets into TensorFlow TFRecordDataset

    :param partition_dir: directory of partition
    :param brats_tfrecords_dir: Directory of TFRecords
    :return: TFRecordDatasets:
        - train_dataset
        - test_dataset
        - validation_dataset
    """
    train_ids = get_training_ids(partition_dir)
    test_ids = get_test_ids(partition_dir)
    validation_ids = get_validation_ids(partition_dir)

    record_map = get_record_id_map(brats_tfrecords_dir)
    train_dataset = get_tfrecord_dataset(train_ids, record_map)
    test_dataset = get_tfrecord_dataset(test_ids, record_map)
    validation_dataset = get_tfrecord_dataset(validation_ids, record_map)

    train_dataset.map(_parse_function)
    train_dataset.map(_reshape_fn)

    return train_dataset, test_dataset, validation_dataset


def get_tfrecord_dataset(patient_ids, record_map):
    """
    Makes a single TFRecordDataset containing the specified IDs

    :param patient_ids: The patient IDs to put in
    :param record_map: Mapping from patient_id to TFRecord filename
    :return: TFRecordDataset containing the TFRecords specified
    """
    filenames = []
    for patient_id in patient_ids:
        filenames.append(record_map[patient_id])
    return tf.data.TFRecordDataset(filenames=filenames,
                                   compression_type=tf.python_io.TFRecordCompressionType.GZIP)


def get_record_id_map(brats_tfrecords_dir):
    """
    Get the mapping from patient_id --> TFRecord file

    :param brats_tfrecords_dir: directory containing all TFRecords
    :return: Dictionary mapping patient_id to TFRecord file
    """
    tfrecord_filenames = os.listdir(brats_tfrecords_dir)
    id_record_map = {}
    for file_name in tfrecord_filenames:
        patient_id = get_id_of_TFRecord(file_name)
        id_record_map[patient_id] = os.path.join(brats_tfrecords_dir, file_name)
    return id_record_map

