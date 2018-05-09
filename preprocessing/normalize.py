#!/usr/bin/env python
"""
File: normalize.py
Date: 5/8/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import sys
import argparse
import multiprocessing as mp

import BraTS
from BraTS.structure import *
from preprocessing.normalization import normalize_patient_images

import logging


pool_size = 8
logger = None

def normalize_brats(brats_root, year, output_directory):
    brats = BraTS.DataSet(brats_root=brats_root, year=year)

    train_corrected = get_brats_subset_directory(output_directory, DataSubsetType.train)
    hgg_corrected = get_brats_subset_directory(output_directory, DataSubsetType.hgg)
    lgg_corrected = get_brats_subset_directory(output_directory, DataSubsetType.lgg)
    validation_corrected = get_brats_subset_directory(output_directory, DataSubsetType.validation)

    # Make the directories
    for directory in (train_corrected, hgg_corrected, lgg_corrected, validation_corrected):
        try:
            os.mkdir(directory)
        except FileExistsError:
            logger.debug("Directory exists: %s" % directory)

    pool = mp.Pool(pool_size)

    # Convert each of the sets
    for patient_set in (brats.hgg, brats.lgg, brats.validation):
        if patient_set is None:
            continue  # Missing data set (e.g. validation is not present)

        logger.info("Processing set: %s" % patient_set.type)

        # Make a list of original-dir -> new-dir outputs
        arg_list = []
        for patient_id in patient_set.ids:
            original_dir = brats.train.directory_map[patient_id]
            new_dir = os.path.join(get_brats_subset_directory(output_directory, patient_set.type), patient_id)
            arg_list.append((original_dir, new_dir, patient_id))

        # Do the conversion in parallel
        pool.map(convert_wrapper, arg_list)

# For converting in parallel
def convert_wrapper(args):
    orig_dir, new_dir, patient_id = args
    try:
        os.mkdir(new_dir)
    except FileExistsError:
        pass
    logger.debug("Processing: %s" % patient_id)
    normalize_patient_images(*(orig_dir, new_dir))


def parse_args():
    """
    Parse command line arguments

    :return: An argparse object containing parsed arguments
    """
    parser = argparse.ArgumentParser(description="Normalize the BraTS data set",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_options = parser.add_argument_group("Input")
    input_options.add_argument('--brats', required=True, help="BraTS root data set directory")
    input_options.add_argument('--year', required=True, type=int, default=2018, help="BraTS year")

    output_options = parser.add_argument_group("Output")
    output_options.add_argument('--output', required=True, help="Output directory of normalized data set")

    general_options_group = parser.add_argument_group("General")
    general_options_group.add_argument("--pool-size", type=int, default=8, help="Size of worker pool")

    logging_options_group = parser.add_argument_group("Logging")
    logging_options_group.add_argument('--log', dest="log_level", default="WARNING", help="Logging level")
    logging_options_group.add_argument('--log-file', default="normalize.log", help="Log file")

    args = parser.parse_args()

    # Setup the logger
    global logger
    logger = logging.getLogger('root')

    # Logging level configuration
    log_level = getattr(logging, args.log_level.upper())
    if not isinstance(log_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
    logger.setLevel(log_level)

    log_formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')

    # For the log file...
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # For the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
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

    normalize_brats(brats_root, args.year, output_dir)


if __name__ == "__main__":
    main()
