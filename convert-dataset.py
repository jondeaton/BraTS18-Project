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

logger = logging.getLogger()

pool_size = 8


def transform_patient_shell(args):
    transform_patient(*args)


def transform_patient(brats_root, patient_id, output_directory):

    BraTS.set_root(brats_root)
    brats = BraTS.DataSet(year=2018)

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

    # Write it to file
    tf_record_filename = os.path.join(output_directory, "%s.tfrecord" % patient_id)
    with tf.python_io.TFRecordWriter(tf_record_filename) as writer:
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


def transform_brats(brats_root, output_dir, num_test=40, num_validation=40):

    BraTS.set_root(brats_root)
    brats = BraTS.DataSet(year=2018)

    ids = brats.train.ids
    shuffle(ids)

    # Split up the patient IDs into test, validation and train
    test_ids = ids[:num_test]
    validation_ids = ids[num_test:(num_test + num_validation)]
    train_ids = ids[(num_test + num_validation):]

    train_records = os.path.join(output_dir, "train_records")
    test_records = os.path.join(output_dir, "test_records")
    validation_records = os.path.join(output_dir, "validation_records")

    write_dataset(brats_root, train_ids, train_records)
    write_dataset(brats_root, test_ids, test_records)
    write_dataset(brats_root, validation_ids, validation_records)


def parse_args():
    """
    Parse command line arguments

    :return: An argparse object containing parsed arguments
    """

    parser = argparse.ArgumentParser(description="Train tumor segmentation model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    io_options_group = parser.add_argument_group("I/O")
    io_options_group.add_argument('--brats', help="BraTS root dataset directory")
    io_options_group.add_argument('--output', help="Output directoyr of dataset")

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

    transform_brats(brats_root, output_dir,
                    num_test=args.test,
                    num_validation=args.validation)


if __name__ == "__main__":
    main()
