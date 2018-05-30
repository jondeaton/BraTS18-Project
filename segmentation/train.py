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

from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import TensorBoard, ModelCheckpoint

import BraTS
from BraTS.Patient import load_patient_data
from BraTS.modalities import mri_shape, seg_shape
from preprocessing.partitions import get_training_ids, get_test_ids

import preprocessing
from preprocessing.load_tfrecords import *

from segmentation.model_UNet3D import UNet3D
from segmentation.metrics import dice_coefficient_loss, dice_coefficient
from segmentation.config import Configuration
from segmentation.visualization import TrainValTensorBoard

from augmentation.augmentation import blur, random_flip, add_noise

logger = logging.getLogger()


def make_generator(patient_ids, augment=False):
    patient_ids = list(patient_ids)
    brats = BraTS.DataSet(brats_root=config.brats_directory, year=2018)
    mri = np.empty((1,) + mri_shape)
    seg = np.empty((1, 1) + seg_shape)

    while True:
        shuffle(patient_ids)
        for patient_id in patient_ids:
            patient = brats.train.patient(patient_id)
            _mri, _seg = patient.mri, patient.seg
            _seg[_seg >= 1] = 1

            yield fix_dims(_mri, _seg, mri, seg)
            if augment:
                yield fix_dims(*random_flip(_mri, _seg), mri, seg)
                yield fix_dims(*add_noise(_mri, _seg), mri, seg)
                yield fix_dims(*blur(_mri, _seg), mri, seg)
            brats.train.drop_cache()


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

    record_map = get_record_id_map("/Users/CamBackes/Documents/Academic/Courses/CS_230/Dataset/BraTS_TFRecords/BraTS18/training")
    
    #train_dataset = get_tfrecord_dataset(patient_ids, record_map)
    
    def _parse_function(serialized):
        mri = tf.FixedLenFeature([], tf.string)
        seg = tf.FixedLenFeature([], tf.int64)

        features = {'train/mri': mri, 'train/seg': seg}

        parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)
        
        # Get the image as raw bytes and decode into tensor
        # ** may need to alter image_shape
        mri_shape = tf.stack([4, 240, 240, 155])
        mri_raw = parsed_example['train/mri']
        mri_T  = tf.decode_raw(mri_raw, tf.uint8)
        mri_T = tf.cast(mri_T, tf.float32)

        #reshape and standardize
        mri_T = tf.reshape(mri_T, mri_shape)
        #mri_T = tf.per_image_standardization(mri_T)
        
        #parse ground truth seg
        seg_raw = parsed_example['train/seg']
        #seg_T = tf.decode_raw(seg_raw, tf.uint8)
        seg_T = tf.cast(seg_raw, tf.float32)
        
        # return zipped dictionary containing mri and seg
        tensor_dict = dict(zip(input_name, mri_T)), seg_T
        return tensor_dict
    
    #get dataset
    #train_dataset = train_dataset.map(_parse_function)
    
    while True:
        shuffle(patient_ids)
        for patient_id in patient_ids:
            patient_dir = record_map[patient_id]
            train_dataset = tf.data.TFRecordDataset(filenames=patient_dir)
            train_dataset = train_dataset.map(_parse_function)
            train_dataset = train_dataset.batch(batch_size)  # Batch size to use
            iterator = train_dataset.make_one_shot_iterator()
            _mri, _seg = iterator.get_next()
            _seg[_seg >= 1] = 1

            yield fix_dims(_mri, _seg, mri, seg)
            # yield fix_dims(*random_flip(_mri, _seg), mri, seg)
            # yield fix_dims(*add_noise(_mri, _seg), mri, seg)
            # yield fix_dims(*blur(_mri, _seg), mri, seg)


class histogram_Callback(Callback):

    def __init__(self, model, file="weights.csv", **kwargs):
        self.model = model
        self.file = file
        super(Callback, self).__init__(**kwargs)

    def  on_batch_end(self, batch, logs=None):
        logs = logs or {}
        with open(self.file, 'w') as f:
            f.write("%d\t" % batch)
            for layer in self.model.layers:
                weights, biases = self.model.layers[0].get_weights()

                weights_mean = np.mean(weights)
                weights_std = np.std(weights)

                bias_mean = np.mean(biases)
                bias_std = np.std(biases)

                f.write("layer: %s, %s %s %s %s\t" % (layer,
                        weights_mean, weights_std, bias_mean, bias_std))
            f.write("\n")


def train(model):
    """
    Trains a model

    :param model: The Keras model to train
    :param test_data: Test/dev set
    :return: None
    """

    metrics = [dice_coefficient]
    model.compile(optimizer=Adam(lr=config.learning_rate),
                  loss=dice_coefficient_loss,
                  metrics=metrics)

    checkpoint_callback = ModelCheckpoint(config.model_file,
                                          save_best_only=True)

    tb_callback = TrainValTensorBoard(log_dir=config.tensorboard_dir,
                                      histogram_freq=0,
                                      write_graph=True,
                                      write_images=True)

    callbacks = [tb_callback, checkpoint_callback]
    model.fit_generator(generator=training_generator(),
                        steps_per_epoch=285,
                        epochs=config.num_epochs,
                        verbose=1,
                        validation_data=make_generator(get_test_ids(), augment=False),
                        nb_val_samples=40,
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

    logger.info("Initiating training.")
    logger.debug("TensorBoard Directory: %s" % config.tensorboard_dir)
    logger.debug("Model save file: %s" % config.model_file)
    logger.debug("Learning rate: %s" % config.learning_rate)
    logger.debug("Num epochs: %s" % config.num_epochs)
    logger.debug("Mini-batch size: %s" % config.mini_batch_size)

    train(model)

    logger.debug("Exiting.")


if __name__ == "__main__":
    main()
