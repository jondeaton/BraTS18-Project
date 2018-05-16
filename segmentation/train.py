#!/usr/bin/env python
"""
File: train
Date: 5/1/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging

import numpy as np
from random import shuffle

from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import TensorBoard, ModelCheckpoint

import BraTS
from BraTS.Patient import load_patient_data
from BraTS.modalities import mri_shape, seg_shape
from preprocessing.partitions import get_training_ids, get_test_ids

from segmentation.model_UNet3D import UNet3D
from segmentation.metrics import dice_coefficient_loss, dice_coefficient
from segmentation.config import Configuration

from augmentation.augmentation import blur, random_flip, add_noise

logger = logging.getLogger()


def get_test_data():
    """
    Loads data to test the model on

    :return:
    """
    test_ids = get_test_ids()
    num_test = len(test_ids)

    brats = BraTS.DataSet(brats_root=config.brats_directory, year=2018)

    test_mris = np.empty((num_test,) + mri_shape)
    test_segs = np.empty((num_test, 1) + seg_shape)
    for i, patient_id in enumerate(test_ids):
        test_mris[i] = brats.train.patient(patient_id).mri
        test_segs[i, 0] = brats.train.patient(patient_id).seg
    return test_mris, test_segs

def fix_dims(mri, seg, out_mri, out_seg):
    out_mri[0] = mri
    out_seg[0, 0] = seg
    return out_mri, out_seg

def training_generator():
    """
    Generates training examples

    :return: Yields a single MRI and segmentation pair
    """
    brats = BraTS.DataSet(brats_root=config.brats_directory, year=2018)
    patient_ids = list(get_training_ids())

    mri = np.empty((1,) + mri_shape)
    seg = np.empty((1, 1) + seg_shape)
    
    while True:
        shuffle(patient_ids)
        for patient_id in patient_ids:
            patient_dir = brats.train.directory_map[patient_id]
            _mri, _seg = load_patient_data(patient_dir)
            _seg[_seg >= 1] = 1

            yield fix_dims(_mri, _seg, mri, seg)
            yield fix_dims(*random_flip(_mri, _seg), mri, seg)
            yield fix_dims(*add_noise(mri, seg), mri, seg)
            yield fix_dims(*blur(mri, seg), mri, seg)


def train(model, test_data):
    """
    Trains a model

    :param model: The Keras model to train
    :param test_data: Test/dev set
    :return: None
    """

    metrics = [dice_coefficient]
    model.compile(optimizer=Adam(lr=config.learning_rate),
                  loss=binary_crossentropy,
                  metrics=metrics)

    checkpoint_callback = ModelCheckpoint(config.model_file,
                                          save_best_only=True)

    tb_callback = TensorBoard(log_dir=config.tensorboard_dir,
                              histogram_freq=1,
                              write_graph=True,
                              write_images=True)


    callbacks = [tb_callback, checkpoint_callback]
    model.fit_generator(generator=training_generator(),
                        steps_per_epoch=205,
                        epochs=config.num_epochs,
                        verbose=1,
                        validation_data=test_data,
                        callbacks=callbacks)

def parse_args():
    """
    Parse the command line options for this file
    :return: An argparse object containing parsed arguments
    """

    parser = argparse.ArgumentParser(description="Train the tumor segmentation model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    info_options = parser.add_argument_group("Info")
    info_options.add_argument("--job-dir", default=None, help="Job directory")
    info_options.add_argument("--job-name", default="BraTS", help="Job name")
    info_options.add_argument("-gcs", "--google-cloud", action='store_true', help="Running in Google Cloud")
    info_options.add_argument("--config", help="Config file.")

    input_options = parser.add_argument_group("Input")
    input_options.add_argument('--brats', help="BraTS root data set directory")
    input_options.add_argument('--year', type=int, default=2018, help="BraTS year")

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("--model-file", help="File to save trained model in")

    tensorboard_options = parser.add_argument_group("TensorBoard")
    tensorboard_options.add_argument("--tensorboard", help="TensorBoard directory")

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

    # For the log file...
    if not args.google_cloud:
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

    global config
    if args.config is not None:
        config = Configuration(config_file=args.config)
    else:
        config = Configuration()

    if args.google_cloud:
        logger.info("Running on Google Cloud.")

    logger.debug("BraTS data set directory: %s" % config.brats_directory)

    logger.info("Defining model.")
    model = UNet3D(mri_shape)

    logger.info("Creating test data...")
    test_data = get_test_data()

    logger.info("Initiating training.")
    logger.debug("TensorBoard Directory: %s" % config.tensorboard_dir)
    logger.debug("Model save file: %s" % config.model_file)
    logger.debug("Learning rate: %s" % config.learning_rate)
    logger.debug("Num epochs: %s" % config.num_epochs)
    logger.debug("Mini-batch size: %s" % config.mini_batch_size)

    train(model, test_data)

    logger.debug("Exiting.")


if __name__ == "__main__":
    main()
