#!/usr/bin/env python
"""
File: evaluate.py
Date: 6/3/18 
Author: Jon Deaton (jdeaton@stanford.edu)

--------------------------------------------

Evaluates a BraTS segmentation model.

"""

import sys
import argparse
import logging

import numpy as np
import tensorflow as tf

import BraTS
from segmentation.config import Configuration
from preprocessing.partitions import get_all_partition_ids

logger = logging.getLogger()


def get_prediction(sess, input, output, mri):
    return sess.run(output, feed_dict={input: mri})

def make_dice_histogram(sess, input, output, patient_ids):
    brats = BraTS.DataSet(data_set_dir=config.brats_directory, year=2018)

    for id in patient_ids:
        patient = brats.train.patient(id)


def hitograms():
    train_ids, test_ids, validation_ids = get_all_partition_ids()

def evaluate(model_file):
    tf.reset_default_graph()

    with tf.Session() as sess:

        logger.info("Restoring model...")
        saver = tf.train.import_meta_graph(model_file)
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        logger.info("Model restored.")

        graph = tf.get_default_graph()

        input = graph.get_tensor_by_name("input")
        output = graph.get_tensor_by_name("output")


def main():
    args = parse_args()

    global config
    config = Configuration(args.config)

    evaluate(args.model_file)


def parse_args():
    """
    Parse the command line options for this file
    :return: An argparse object containing parsed arguments
    """

    parser = argparse.ArgumentParser(description="Train the tumor segmentation model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    info_options = parser.add_argument_group("Info")
    info_options.add_argument("--config", required=True, type=str, help="Configuration file")
    info_options.add_argument("-params", "--params", type=str, help="Hyperparameters json file")

    input_options = parser.add_argument_group("Input")
    input_options.add_argument('--brats', help="BraTS root data set directory")

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("--model-file", help="File to save trained model in")

    logging_options = parser.add_argument_group("Logging")
    logging_options.add_argument('--log', dest="log_level", default="DEBUG", help="Logging level")
    logging_options.add_argument('--log-file', default="model.log", help="Log file")

    args = parser.parse_args()

    # Setup the logger
    global logger
    logger = logging.getLogger('root')

    # Logging level configuration
    log_level = getattr(logging, args.log_level.upper())
    if not isinstance(log_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)

    log_formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')

    # Log to file if not on google cloud
    if not args.google_cloud:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    # For the console
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(log_level)

    return args



if __name__ == "__main__":
    main()
