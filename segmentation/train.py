#!/usr/bin/env python
"""
File: train
Date: 5/1/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import argparse
import logging
import time

import numpy as np
import tensorflow as tf

from tensorflow import keras 
from keras import Input
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D)
from keras.models import Model


from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
import io


import BraTS
from segmentation.partitions import load_datasets
from BraTS.modalities import mri_shape, image_shape, seg_shape


# Global Variables
tensorboard_dir = None
save_file = None
learning_rate = None
num_epochs = None
mini_batch_size = None
seed = 0

logger = logging.getLogger()


def model(X, Y):
    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable("kernel", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        Z1 = tf.nn.conv2d(X, kernel, strides=[1, 1, 1, 1], padding='SAME')
        A1 = tf.nn.relu(Z1)
        P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable("kernel", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        Z2 = tf.nn.conv2d(P1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        A2 = tf.nn.relu(Z2)
        P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

    P2_flat = tf.contrib.layers.flatten(P2)
    return None

<<<<<<< HEAD
def train(train_set, test_set):
    # todo!
    pass
=======

def train(train_dataset, test_dataset):
    input_shape = (None,) + mri_shape
    output_shape = (None,) + seg_shape
    X = tf.placeholder(tf.float32, shape=input_shape)
    Y = tf.placeholder(tf.float32, shape=output_shape)
    pred_seg = model(X, Y)
>>>>>>> 1329bef0ba02d5cc40b0533de327533cd275dd6a

def model(input_shape):
    '''Create 3D cnn model with parameters specified
        return keras Model instance of Unet'''

    X_input = Input(input_shape)

    # Unet applied to X_input
    X_input = Conv3D(filter = 32, kernel_size =(3, 3, 3))(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    return model


def parse_args():
    """
    Parse the command line options for this file
    :return: An argparse object containing parsed arguments
    """

    parser = argparse.ArgumentParser(description="Train tumor segmentation model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    info_options = parser.add_argument_group("Info")
    info_options.add_argument("--job-dir", default=None, help="Job directory")
    info_options.add_argument("--job-name", default="signs", help="Job name")
    info_options.add_argument("-gcs", "--google-cloud", action='store_true', help="Running in Google Cloud")
    info_options.add_argument("-aws", "--aws", action="store_true", help="Running in Amazon Web Services")

    input_options = parser.add_argument_group("Input")
    input_options.add_argument('--brats', help="BraTS root dataset directory")
    input_options.add_argument('--records', help="TFRecords for data set directory")

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("--save-file", help="File to save trained model in")

    tensorboard_options = parser.add_argument_group("TensorBoard")
    tensorboard_options.add_argument("--tensorboard", help="TensorBoard directory")

    hyper_params = parser.add_argument_group("Hyper-Parameters")
    hyper_params.add_argument("-l", "--learning-rate", type=float, default=0.0001, help="Learning rate")
    hyper_params.add_argument("-e", "--epochs", type=int, default=1500, help="Number of training epochs")
    hyper_params.add_argument("-mb", "--mini-batch", type=int, default=128, help="Mini-batch size")

    logging_options = parser.add_argument_group("Logging")
    logging_options.add_argument('--log', dest="log_level", default="WARNING", help="Logging level")
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

    if args.google_cloud:
        logger.info("Running on Google Cloud.")
    elif args.aws:
        logger.info("Running in Amazon Web Services.")
    else:
        logger.debug("Running locally.")

    global tensorboard_dir, records_directory, save_file, brats_directory
    brats_directory = os.path.expanduser(args.brats)
    records_directory = os.path.expanduser(args.records)
    tensorboard_dir = args.tensorboard
    save_file = args.save_file

    logger.debug("Data-set Directory: %s" % brats_directory)
    logger.debug("TensorBoard Directory: %s" % tensorboard_dir)
    logger.debug("Save file: %s" % save_file)

    global learning_rate, num_epochs, mini_batch_size
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    mini_batch_size = args.mini_batch

    logger.info("Learning rate: %s" % learning_rate)
    logger.info("Num epochs: %s" % num_epochs)
    logger.info("Mini-batch size: %s" % mini_batch_size)

    logger.info("Loading BraTS data-set...")
    train_dataset, test_dataset, validation_dataset = load_datasets(records_directory)
    logger.info("Data-set loaded.")

    train(train_dataset, test_dataset)



if __name__ == "__main__":
    main()
