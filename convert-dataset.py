#!/usr/bin/env python
"""
File: convert-dataset
Date: 5/5/18 
Author: Jon Deaton (jdeaton@stanford.edu)

Converts the BraTS dataset into TensorFlow's native TFRecord format
so that it can be loaded in using tf.data.TFRecordDataset during training.

--------------------------------------------------------------------------

Note when I tried using this on the BraTS 2018 dataset, it ended up
inflating the dataset to be 50 GB so this is probably not a viable solution

Instead we are exploring creating a tf.Dataset from a genreator
built on top of the BraTS loader module

Better solution: use GZIP format for the TFRecords and it stays like 2GB
"""

import os
import sys
import argparse
import logging
import BraTS

from random import shuffle
import tensorflow as tf
import numpy as np
import multiprocessing as mp

from segmentation.partitions import *
from segmentation.partitioning import *

logger = logging.getLogger()
pool_size = 8  # Pool of worker processes


def transform_patient_shell(args):
    transform_patient(*args)


def transform_patient(brats, patient_id, output_directory):

    logger.info("Transforming: %s" % patient_id)

    # Load the patient data
    patient = brats.train.patient(patient_id)

    # Gotta compress it to a 1D array first :/
    mri_list = np.reshape(patient.mri, newshape=(np.prod(patient.mri.shape, )))
    seg_list = np.reshape(patient.seg, newshape=(np.prod(patient.seg.shape, )))

    # Put it into a numpy feature
    mri = tf.train.Feature(float_list=tf.train.FloatList(value=mri_list))
    seg = tf.train.Feature(int64_list=tf.train.Int64List(value=seg_list))

    # Convert it into some TensorFlow
    feature = {'train/mri': mri, 'train/seg': seg}
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Write it to file (compressed)
    tf_record_filename = os.path.join(output_directory, "%s.tfrecord.gzip" % patient_id)
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    with tf.python_io.TFRecordWriter(tf_record_filename, options=options) as writer:
        writer.write(example.SerializeToString())


def write_dataset(brats_root, ids, output_directory):
    if not os.path.isdir(output_directory):
        logger.debug("Creating output directory: %s" % output_directory)
        try:
            os.mkdir(output_directory)
        except FileExistsError:
            logger.debug("Output directory exists: %s" % output_directory)

    pool = mp.Pool(pool_size)
    arg_list = [(brats_root, patient_id, output_directory) for patient_id in ids]
    pool.map(transform_patient_shell, arg_list)


def transform_brats(brats_root, output_dir, year, train_ids, test_ids, validation_ids):
    brats = BraTS.DataSet(brats_root=brats_root, year=year)

    train_records_dir = os.path.join(output_dir, train_records_dirname)
    test_records_dir = os.path.join(output_dir, test_records_dirname)
    validation_records_dir = os.path.join(output_dir, validation_records_dirname)

    write_dataset(brats, train_ids, train_records_dir)
    write_dataset(brats, test_ids, test_records_dir)
    write_dataset(brats, validation_ids, validation_records_dir)


def parse_args():
    """
    Parse command line arguments

    :return: An argparse object containing parsed arguments
    """
    parser = argparse.ArgumentParser(description="Convert a BraTS Dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    io_options_group = parser.add_argument_group("I/O")
    io_options_group.add_argument('--brats', help="BraTS root dataset directory")
    io_options_group.add_argument('--output', help="Output directoy of dataset")
    io_options_group.add_argument('--partition-dir', default=default_partition_store, help="Directory of partitions")
    io_options_group.add_argument('--year', type=int, default=default_brats_year, help="BraTS year")

    sets_options_group = parser.add_argument_group("Data set")
    sets_options_group.add_argument("--test", type=int, default=40, help="Size of training set")
    sets_options_group.add_argument("--validation", type=int, default=40, help="Size of validation set")

    general_options_group = parser.add_argument_group("General")
    general_options_group.add_argument("--pool-size", type=int, default=8, help="Size of worker pool")

    logging_options_group = parser.add_argument_group("Logging")
    logging_options_group.add_argument('--log', dest="log_level", default="WARNING", help="Logging level")
    logging_options_group.add_argument('--log-file', default="model.log", help="Log file")

    args = parser.parse_args()

    # Setup the logger
    global logger
    logger = logging.getLogger('root')

    # Logging level configuration
    log_level = getattr(logging, args.log_level.upper())
    if not isinstance(log_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)

    log_formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')

    # For the log file...
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # For the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(log_level)

    return args


def main():
    args = parse_args()

    brats_root = os.path.expanduser(args.brats)
    output_dir = os.path.expanduser(args.output)

    global pool_size
    pool_size = args.pool_size

    if not os.path.isdir(brats_root):
        raise FileNotFoundError(brats_root)

    logger.debug("BraTS root: %s" % brats_root)

    if not os.path.exists(output_dir):
        logging.debug("Creating output directory: %s" % output_dir)
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            logger.debug("Output directory exists.")
    else:
        logger.debug("Output directory: %s" % output_dir)

    logger.debug("Number of test examples: %d" % args.test)
    logger.info("Number of validation examples: %d" % args.validation)

    train_ids = get_training_ids(args.partition_dir)
    test_ids = get_test_ids(args.partition_dir)
    validation_ids = get_validation_ids(args.partition_dir)

    transform_brats(brats_root, output_dir, args.year,
                    train_ids, test_ids, validation_ids)


if __name__ == "__main__":
    main()
