#!/usr/bin/env python
"""
File: normalize.py
Date: 5/8/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import argparse
import logging

import BraTS
from preprocessing.normalization import *

def setup_output (output_directory):
    assert isinstance(output_directory, str)

    train_corrected = os.path.join(output_directory, "training")
    validation_corrected = os.path.join(output_directory, "validation")
    


def normalize_brats(brats_root, year):

    brats = BraTS.DataSet(brats_root=brats_root, year=year)




def parse_args():
    """
    Parse command line arguments

    :return: An argparse object containing parsed arguments
    """
    parser = argparse.ArgumentParser(description="Pre-process BraTS dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_options = parser.add_argument_group("Input")
    input_options.add_argument('--brats', help="BraTS root dataset directory")
    input_options.add_argument('--year', type=int, default=2018, help="BraTS year")

    output_options = parser.add_argument_group("Output")
    output_options.add_argument('--output', help="Output directory of normalized data set")


    general_options_group = parser.add_argument_group("General")
    general_options_group.add_argument("--pool-size", type=int, default=8, help="Size of worker pool")

    logging_options_group = parser.add_argument_group("Logging")
    logging_options_group.add_argument('--log', dest="log_level", default="WARNING", help="Logging level")
    logging_options_group.add_argument('--log-file', default="convert.log", help="Log file")

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

    logger.debug("Number of test examples: %d" % args.test)
    logger.info("Number of validation examples: %d" % args.validation)

    brats = BraTS.DataSet(brats_root=brats_root, year=args.year)
    transform_brats(brats_root, args.year, output_dir, brats.train.ids)


if __name__ == "__main__":
    main()
