"""

This code isn't useful anymore but I thought I might keep it around.

"""


import os
import tensorflow as tf
from preprocessing.partitioning import default_partition_store, get_training_ids, get_test_ids, get_validation_ids
from preprocessing.records import get_id_of_TFRecord

from BraTS.Patient import load_patient_data
from BraTS.modalities import get_modality_file, Modality, mri_shape, seg_shape


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