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
from BraTS.Patient import Patient
from segmentation.config import Configuration
from preprocessing.partitions import get_all_partition_ids
import matplotlib.pyplot as plt
import random

logger = logging.getLogger()


def dice_coefficient(pred, truth, smooth=0.02):
    _pred = np.ravel(pred)
    _truth = np.ravel(truth)
    intersection = np.logical_and(_pred, _truth)
    return (2 * intersection.sum() + smooth) / (pred.sum() + truth.sum() + smooth)


def to_single_class(seg, threshold):
    _seg = np.copy(seg)
    _seg[_seg > 0] = 1
    return _seg


def make_dice_histogram(dice_coefficients, filename):
    # todo: make histogram and save it...
    pass

def get_tumor_range(patient):
    assert isinstance(patient, Patient)
    tumor_range = list()
    for i in range(patient.seg.shape[3]):
        if np.sum(patient.seg[:,:,i]) != 0:
            tumor_range.append(i)
    return tumor_range


def make_image(patient, out):
    assert isinstance(patient, Patient)
    assert isinstance(out, np.ndarray)
    tumor_range = get_tumor_range(patient)

    random.shuffle(tumor_range)

    for slice_index in tumor_range[10:]:
        pass
        # coronal_slice = patient.t1[:,:,0]
        # fig, axarr = plt.subplots(1, 2)
        #
        # axarr[0].imshow(patient.)
        # axarr[0].set_title("Original")
        #
        #
        # axarr[1].imshow(yTilde[0:N].reshape((40, 40)))
        # axarr[1].set_title("Original Noisy")
        # fig.savefig("originals_%.2f.png" % epsilon)
        # todo: fix


def make_histograms_and_images(sess, input, output, patient_ids):

    brats = BraTS.DataSet(data_set_dir=config.brats_directory, year=2018)

    dice_coefficients = list()

    for id in patient_ids:
        patient = brats.train.patient(id)
        out = sess.run(output, feed_dict={input: patient.mri})
        pred = to_single_class(out)
        truth = to_single_class(patient.seg)

        dice = dice_coefficient(pred, truth)
        logger.info("Patient: %d, dice coefficient: %s" % (id, dice))
        dice_coefficients.append(dice)
        make_image(patient, out)

    make_dice_histogram(dice_coefficients, "dice_hist.png")




def evaluate(sess, input, output):
    assert isinstance(sess, tf.Session)
    assert isinstance(input, tf.Tensor)
    assert isinstance(output, tf.Tensor)

    train_ids, test_ids, validation_ids = get_all_partition_ids()

    make_histograms_and_images(sess, input, output, train_ids)
    make_histograms_and_images(sess, input, output, test_ids)
    make_histograms_and_images(sess, input, output, validation_ids)


def restore_and_evaluate(model_file):
    tf.reset_default_graph()

    with tf.Session() as sess:

        logger.info("Restoring model: %s" % model_file)
        saver = tf.train.import_meta_graph(model_file)
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        logger.info("Model restored.")

        graph = tf.get_default_graph()

        input = graph.get_tensor_by_name("input")
        output = graph.get_tensor_by_name("output")

        logger.info("Evaluating mode...")
        evaluate(sess, input, output)


def main():
    args = parse_args()

    global config
    config = Configuration(args.config)

    restore_and_evaluate(args.model_file)


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

    logger.setLevel(log_level)

    return args



if __name__ == "__main__":
    main()
