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
import multiprocessing as mp

import BraTS
from preprocessing.partitions import make_tfrecord

logger = logging.getLogger()
pool_size = 8  # Pool of worker processes


def _make_tfrecord_shell(args):
    make_tfrecord(*args)


def make_tfrecords(brats_root, year, output_directory, ids, sequential=False):
    if not os.path.isdir(output_directory):
        logger.debug("Creating output directory: %s" % output_directory)
        try:
            os.mkdir(output_directory)
        except FileExistsError:
            logger.debug("Output directory exists: %s" % output_directory)

    if sequential:
        for patient_id in ids:
            make_tfrecord(brats_root, year, output_directory, patient_id)
    else:
        pool = mp.Pool(pool_size)
        arg_list = [(brats_root, year, output_directory, patient_id) for patient_id in ids]
        pool.map(_make_tfrecord_shell, arg_list)


def parse_args():
    """
    Parse command line arguments

    :return: An argparse object containing parsed arguments
    """
    parser = argparse.ArgumentParser(description="Convert a BraTS Data-set",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    io_options_group = parser.add_argument_group("I/O")
    io_options_group.add_argument('--brats', required=True, help="BraTS root dataset directory")
    io_options_group.add_argument('--year', required=True, type=int, help="BraTS year")
    io_options_group.add_argument('--output', required=True, help="Output directory of dataset")

    general_options_group = parser.add_argument_group("General")
    general_options_group.add_argument("--pool-size", type=int, default=8, help="Size of worker process pool")
    general_options_group.add_argument("-seq", "--sequential", action='store_true', help="Sequentially")

    logging_options_group = parser.add_argument_group("Logging")
    logging_options_group.add_argument('--log', dest="log_level", default="WARNING", help="Logging level")
    logging_options_group.add_argument('--log-file', default="createTFRecords.log", help="Log file")

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
        logger.debug("Creating output directory: %s" % output_dir)
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            logger.debug("Output directory exists.")
    else:
        logger.debug("Output directory: %s" % output_dir)

    brats = BraTS.DataSet(brats_root=brats_root, year=args.year)
    make_tfrecords(brats_root, args.year, output_dir, brats.train.ids, sequential=args.sequential)


if __name__ == "__main__":
    main()
