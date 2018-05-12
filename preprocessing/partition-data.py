#!/usr/bin/env python
"""
File: BraTS_partitions
Date: 5/5/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import sys
import argparse
import logging

from preprocessing.partitioning import *


def parse_args():
    """
    Parse command line arguments

    :return: An argparse object containing parsed arguments
    """

    parser = argparse.ArgumentParser(description="Train tumor segmentation model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    io_options_group = parser.add_argument_group("I/O")
    io_options_group.add_argument('--brats', required=True,  help="BraTS data root directory")
    io_options_group.add_argument('--output', required=False, default=default_partition_store, help="Where the ids are stored")

    sets_options_group = parser.add_argument_group("Partition Set")
    sets_options_group.add_argument("--year", type=int, default=default_brats_year, help="BraTS data year")
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

    logger.debug("Number of test examples: %d" % args.test)
    logger.info("Number of validation examples: %d" % args.validation)

    generate_random_partitioning(brats_root, output_dir, args.year,
                                 num_test=args.test,
                                 num_validation=args.validation)


if __name__ == "__main__":
    main()
