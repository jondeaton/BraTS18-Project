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
import configparser

import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import Input
from keras.models import Model
from keras.layers import Activation, Conv3D, MaxPooling3D
from keras.layers import BatchNormalization, Concatenate, UpSampling3D
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

from segmentation.metrics import dice_coefficient_loss, dice_coefficient

import BraTS
from BraTS.modalities import mri_shape, seg_shape
from preprocessing.partitions import load_datasets
from augmentation.augmentation import augment_training_set
from preprocessing.partitions import get_training_ids, get_test_ids

from random import shuffle

# Global Variables
tensorboard_dir = None
save_file = None
learning_rate = None
num_epochs = None
mini_batch_size = None
seed = 0

logger = logging.getLogger()


def ConvBlockDown(input_layer, num_filters=32):

    strides = (1, 1, 1)
    kernel = (3, 3, 3)

    layer = Conv3D(num_filters,
                   kernel,
                   data_format="channels_first",
                   strides=strides,
                   padding='same')(input_layer)
    layer = BatchNormalization(axis=1)(layer)
    return Activation('relu')(layer)


def ConvBlockUp(input_layer, concat, num_filters=32):
    strides = (1, 1, 1)
    pool_size = (2, 2, 2)
    kernel = (3, 3, 3)

    X = UpSampling3D(size=pool_size)(input_layer)
    X = Concatenate(axis=1)([X, concat])

    X = Conv3D(num_filters,
               kernel,
               data_format="channels_first",
               strides=strides,
               padding='same')(X)

    X = Conv3D(num_filters,
               kernel,
               data_format="channels_first",
               strides=strides,
               padding='same')(X)
    return X


def UNetLevel(input_layer, num_filters):
    X = ConvBlockDown(input_layer, num_filters=num_filters)
    X = ConvBlockDown(X, num_filters=num_filters)
    X = MaxPooling3D((3, 3, 3), name='max_pool')(X)


def UNet3D(input_shape):
    """
    3D UNet Module implemented in Keras

    :param input_shape:
    :return:
    """

    X_input = Input(input_shape)

    pool_size = (2, 2, 2)

    # Unet applied to X_input

    # Level 1
    X = ConvBlockDown(X_input, num_filters=32)
    X = ConvBlockDown(X, num_filters=32)
    level_1 = X
    X = MaxPooling3D(pool_size=pool_size,
                     name='max_pool1')(X)

    # Level 2
    X = ConvBlockDown(X, num_filters=64)
    X = ConvBlockDown(X, num_filters=64)
    level_2 = X
    X = MaxPooling3D(pool_size=pool_size,
                     name='max_pool2')(X)

    # Level 3
    X = ConvBlockDown(X, num_filters=128)
    X = ConvBlockDown(X, num_filters=128)
    level_3 = X
    X = MaxPooling3D(pool_size=pool_size, name='max_pool3')(X)

    # Lowest Level
    X = ConvBlockDown(X, num_filters=128)
    X = ConvBlockDown(X, num_filters=128)

    # Up-convolutions
    X = ConvBlockUp(X, level_3, num_filters=128)
    X = ConvBlockUp(X, level_2, num_filters=64)
    X = ConvBlockUp(X, level_1, num_filters=32)

    # Create model.
    model = Model(inputs=X_input, outputs=X, name='UNet')

    final_convolution = Conv3D(1,
                               (1, 1, 1),
                               data_format="channels_first",
                               strides=(1, 1, 1),
                               padding='same')(X)
    act = Activation("sigmoid")(final_convolution)
    model = Model(inputs=X_input, outputs=act)

    metrics = [dice_coefficient]
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=binary_crossentropy,
                  metrics=metrics)
    return model


def training_generator():
    brats = BraTS.DataSet(brats_root=brats_directory, year=2018)
    patient_ids = list(get_training_ids())
    shuffle(patient_ids)

    mri = np.empty((1,) + mri_shape)
    seg = np.empty((1, 1,) + seg_shape)

    for patient_id in patient_ids:
        patient = brats.train.patient(patient_id)
        mri[0] = patient.mri
        seg[0, 0] = patient.seg
        seg[0, 0][seg[0, 0] >= 1] = 1  # disregard different tumor intensities
        yield mri, seg
        brats.drop_cache()

def validation_generator(brats_directory):
    brats = BraTS.DataSet(brats_root=brats_directory, year=2018)
    patient_ids = get_training_ids()
    shuffle(patient_ids)
    for patient_id in patient_ids:
        yield brats.train.patient(patient_id)


def train(model, validation_data):
    model.fit_generator(generator=training_generator(),
                        steps_per_epoch=1,
                        epochs=num_epochs)


def parse_args():
    """
    Parse the command line options for this file
    :return: An argparse object containing parsed arguments
    """

    parser = argparse.ArgumentParser(description="Train tumor segmentation model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    info_options = parser.add_argument_group("Info")
    info_options.add_argument("--job-dir", default=None, help="Job directory")
    info_options.add_argument("--job-name", default="BraTS", help="Job name")
    info_options.add_argument("-gcs", "--google-cloud", action='store_true', help="Running in Google Cloud")
    info_options.add_argument("-aws", "--aws", action="store_true", help="Running in Amazon Web Services")
    info_options.add_argument("--config", default="train_config.ini", help="Config file.")

    input_options = parser.add_argument_group("Input")
    input_options.add_argument('--brats', help="BraTS root dataset directory")
    input_options.add_argument('--year', type=int, default=2018, help="BraTS year")

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("--save-file", help="File to save trained model in")

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


def get_test_data():
    test_ids = get_test_ids()
    num_test = len(test_ids)

    brats = BraTS.DataSet(brats_root=brats_directory, year=2018)

    test_mris = np.empty((num_test,) + mri_shape)
    test_segs = np.empty((num_test,) + seg_shape)
    for i, patient_id in enumerate(test_ids):
        test_mris[i] = brats.train.patient(patient_id).mri
        test_segs[i] = brats.train.patient(patient_id).seg
    return test_mris, test_segs

def main():
    args = parse_args()

    if args.google_cloud:
        logger.info("Running on Google Cloud.")
    elif args.aws:
        logger.info("Running in Amazon Web Services.")
    else:
        logger.debug("Running locally.")

    global config
    config = configparser.ConfigParser()
    config.read(args.config)

    global tensorboard_dir, save_file, brats_directory
    brats_directory = os.path.expanduser(config["BraTS"]["root"])
    tensorboard_dir = os.path.expanduser(config["TensorFlow"]["tensorboard-dir"])
    save_file = os.path.expanduser(config["Output"]["save-file"])

    logger.debug("BraTS directory: %s" % brats_directory)
    logger.debug("TensorBoard Directory: %s" % tensorboard_dir)
    logger.debug("Save file: %s" % save_file)

    global learning_rate, num_epochs, mini_batch_size
    learning_rate = float(config["Hyperparameters"]["learning-rate"])
    num_epochs = int(config["Hyperparameters"]["epochs"])
    mini_batch_size = int(config["Hyperparameters"]["mini-batch"])

    logger.info("Learning rate: %s" % learning_rate)
    logger.info("Num epochs: %s" % num_epochs)
    logger.info("Mini-batch size: %s" % mini_batch_size)

    logger.info("Defining model.")
    model = UNet3D(mri_shape)

    logger.info("Creating test data...")
    # test_data = get_test_data()

    logger.debug("Initiating training")
    train(model, None)

    logger.debug("Exiting.")


if __name__ == "__main__":
    main()
