#!/usr/bin/env python
"""
File: partitions
Date: 5/8/18 
Author: Jon Deaton (jdeaton@stanford.edu)

-----------------------------------------

Yields the training, testing, and validation tf.data.Dataset

"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io

import BraTS
from BraTS.modalities import get_modality_file, Modality, mri_shape, seg_shape
from preprocessing.partitioning import default_partition_store, get_all_partition_ids
from preprocessing.records import get_id_of_TFRecord
from preprocessing.records import get_TFRecord_filename


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

    def _parse_function(example_proto):
        # For parsing TFRecord files
        keys_to_features = {'mri': tf.FixedLenFeature((mri_shape), tf.float32),
                            'seg': tf.FixedLenFeature((seg_shape), tf.float32)}
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        return parsed_features['mri'], parsed_features['seg']

    # Retrieve the IDs for each of the dataset partitions
    train_ids, test_ids, validation_ids = get_all_partition_ids(partition_dir)

    # Get the mapping from patient ID --> TFRecord file
    record_map = get_record_id_map(brats_tfrecords_dir)

    # Loop through train, test, and validation sets, parsing all TFRecords
    datasets = [None, None, None]
    for i, ids in enumerate((train_ids, test_ids, validation_ids)):
        datasets[i] = get_tfrecord_dataset(ids, record_map)
        datasets[i] = datasets[i].map(_parse_function)
        datasets[i] = datasets[i].map(_reshape_fn)
    return datasets  # list of TFRecordDatasets for tran, test, validation


def get_tfrecord_dataset(patient_ids, record_map):
    """
    Makes a single TFRecordDataset containing the specified IDs

    :param patient_ids: The patient IDs to put in
    :param record_map: Mapping from patient_id to TFRecord filename
    :return: TFRecordDataset containing the TFRecords specified
    """
    filenames = [record_map[id] for id in patient_ids]
    return tf.data.TFRecordDataset(filenames=filenames, compression_type="GZIP")


def get_record_id_map(brats_tfrecords_dir):
    """
    Get the mapping from patient_id --> TFRecord file

    :param brats_tfrecords_dir: directory containing all TFRecords
    :return: Dictionary mapping patient_id to TFRecord file
    """
    tfrecord_filenames = file_io.list_directory(brats_tfrecords_dir)
    id_record_map = {}
    for file_name in tfrecord_filenames:
        patient_id = get_id_of_TFRecord(file_name)
        id_record_map[patient_id] = os.path.join(brats_tfrecords_dir, file_name)
    return id_record_map


def make_tfrecord(brats_root, year, output_directory, patient_id):
    """
    Creates a TFRecord file for a single patient example

    :param brats_root: Root directory of BraTS
    :param year: BraTS dataset year
    :param output_directory: Output directory to store TFRecord file
    :param patient_id: ID of the patient to convert
    :return: None
    """
    # Write it to file (compressed)
    tfrecord_fname = os.path.join(output_directory, get_TFRecord_filename(patient_id))
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    with tf.python_io.TFRecordWriter(tfrecord_fname, options=options) as writer:

        # Load the patient data
        brats = BraTS.DataSet(brats_root=brats_root, year=year)
        patient = brats.train.patient(patient_id)

        # Wrap the data in 5 layers of API calls...
        feature = dict()
        feature['mri'] = tf.train.Feature(float_list=tf.train.FloatList(value=patient.mri.flatten()))
        feature['seg'] = tf.train.Feature(float_list=tf.train.FloatList(value=patient.seg.flatten()))

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        serialized = example.SerializeToString()
        writer.write(serialized)


def _reshape_fn(mri, seg):
    _mri = tf.reshape(mri, (1,) + mri_shape)
    _seg =  tf.reshape(seg, (1, 1,) + seg_shape)
    return _mri, _seg