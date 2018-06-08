#!/usr/bin/env python
"""
File: evaluate.py
Date: 6/3/18 
Author: Jon Deaton (jdeaton@stanford.edu)

--------------------------------------------

Evaluates a BraTS segmentation model.

"""

import os
import sys
import argparse
import logging

import numpy as np
import tensorflow as tf

import BraTS
from BraTS.Patient import Patient
from segmentation.config import Configuration
from preprocessing.partitions import get_all_partition_ids
import random

from BraTS.modalities import mri_shape, seg_shape

import matplotlib.pyplot as plt

logger = logging.getLogger()


def dice_coefficient(pred, truth, smooth=0.02):
    _pred = np.ravel(pred)
    _truth = np.ravel(truth)
    intersection = np.logical_and(_pred, _truth)
    return (2 * intersection.sum() + smooth) / (pred.sum() + truth.sum() + smooth)


def to_single_class(seg, threshold):
    _seg = np.copy(seg)
    _seg[seg >= threshold] = 1
    _seg[seg < threshold] = 0
    return _seg.astype(int)


def make_dice_histogram(dice_coefficients, filename):
    # todo: make histogram and save it...
    pass


def _crop(image):
    image = image[..., 3:]
    return image


def get_tumor_range(patient):
    assert isinstance(patient, Patient)
    tumor_range = list()
    for i in range(patient.seg.shape[3]):
        if np.sum(patient.seg[:, :, i]) != 0:
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


def make_histograms_and_images(get_segmentation, patient_ids, output_dir, name="unnamed"):

    brats = BraTS.DataSet(brats_root=config.brats_directory, year=2018)

    dice_coefficients = list()

    for id in patient_ids:
        patient = brats.train.patient(id)
        mri = np.expand_dims(patient.mri, axis=0)


        out = get_segmentation(_crop(mri))

        pred = to_single_class(out, threshold=1)
        truth = to_single_class(_crop(patient.seg), threshold=0.5)

        dice = dice_coefficient(out, truth)
        logger.info("Patient: %s, dice coefficient: %s" % (id, dice))

        dice_coefficients.append(dice)
        # make_image(patient, out)
    return

    histogram_file = os.path.join(output_dir, "%s_hist.png" % name)
    make_dice_histogram(dice_coefficients, histogram_file)


def evaluate(get_segmentation, output_dir):

    train_ids, test_ids, validation_ids = get_all_partition_ids()

    make_histograms_and_images(get_segmentation, train_ids, output_dir)
    make_histograms_and_images(get_segmentation, test_ids, output_dir)
    make_histograms_and_images(get_segmentation, validation_ids, output_dir)


def restore_and_evaluate(save_path, model_file, output_dir):
    tf.reset_default_graph()

    with tf.Session() as sess:

        # logger.info("Instantiating model...")
        # input = tf.placeholder(shape=(None,4,240,240,152), dtype=tf.float32)
        # seg = tf.placeholder(shape=(None,1,240,240,152) + seg_shape, dtype=tf.float32)
        # output, is_training = UNet.model(input, seg, False, False)

        logger.info("Restoring model: %s" % model_file)
        saver = tf.train.import_meta_graph(model_file)
        saver.restore(sess, tf.train.latest_checkpoint(save_path))

        # saver = tf.train.Saver()
        # saver.restore(sess, model_file)

        logger.info("Model restored.")

        graph = tf.get_default_graph()

        input = graph.get_tensor_by_name("input:0")
        output = graph.get_tensor_by_name("output_1:0")
        is_training = graph.get_tensor_by_name("Placeholder_1:0")

        logger.info("Evaluating mode...")

        # Create a closure that encapsulates this horrible syntax
        # into a function that can be called to simply get
        # a prediction for an input
        def get_segmentation(mri):
            feed_dict = {input: mri, is_training: True}
            return sess.run(output, feed_dict=feed_dict)

        evaluate(get_segmentation, output_dir)


def main():
    args = parse_args()

    global config
    if args.config is not None:
        config = Configuration(args.config)
    else:
        config = Configuration()

    save_path = os.path.expanduser(args.save_path)
    if not os.path.isdir(save_path):
        logger.error("No such save-path directory: %s" % save_path)
        return

    model_file = os.path.join(save_path, args.model)
    if not os.path.exists(model_file):
        logger.error("No such file: %s" % model_file)
        return

    output_dir = os.path.expanduser(args.output)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    restore_and_evaluate(save_path, model_file, output_dir)


def parse_args():
    """
    Parse the command line options for this file
    :return: An argparse object containing parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate the tumor segmentation model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_options = parser.add_argument_group("Input")
    input_options.add_argument("--save-path", required=True, help="Tensorflow save path")
    input_options.add_argument("--model", required=True, help="File to save trained model in")

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("-o", "--output", required=True, help="Output directory to store plots")

    info_options = parser.add_argument_group("Info")
    info_options.add_argument("--config", required=False, type=str, help="Configuration file")
    info_options.add_argument("-params", "--params", type=str, help="Hyperparameters json file")

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

    # For the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(log_level)

    return args


if __name__ == "__main__":
    main()
