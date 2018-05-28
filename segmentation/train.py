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
import datetime

import tensorflow as tf

from preprocessing.partitions import load_tfrecord_datasets
from augmentation.augmentation import augment_training_set
import segmentation.UNet3D as UNet
from segmentation.metrics import dice_coeff, dice_loss
from segmentation.config import Configuration

logger = logging.getLogger()


def _get_job_name():
    now = '{:%Y-%m-%d.%H:%M}'.format(datetime.datetime.now())
    return "%s_lr_%.4f" % (now, config.learning_rate)


def _crop(mri, seg):
    # reshapes images to be (1,4,240,240,152)
    # so that they are easily divisible by powers of two
    size = [-1] * 4
    begin = [0] * 3 + [3]
    _mri = tf.slice(mri, begin=begin, size=size)
    _seg = tf.slice(seg, begin=begin, size=size)
    return _mri, _seg


def create_data_pipeline():
    train_dataset, test_dataset, validation_dataset = load_tfrecord_datasets(config.tfrecords_dir)

    # Crop them to be 240,240,152
    train_dataset = train_dataset.map(_crop)
    test_dataset = test_dataset.map(_crop)
    validation_dataset = validation_dataset.map(_crop)

    # Dataset augmentation
    train_aug = augment_training_set(train_dataset)

    # Shuffle, repeat, batch, prefetch the training dataset
    train_aug = train_aug.shuffle(config.shuffle_buffer_size)
    train_aug = train_aug.batch(config.mini_batch_size)
    train_aug = train_aug.prefetch(buffer_size=config.prefetch_buffer_size)

    # Shuffle/batch test dataset
    test_dataset = test_dataset.shuffle(config.shuffle_buffer_size)
    test_dataset = test_dataset.batch(config.mini_batch_size)

    return train_aug, test_dataset, validation_dataset


def _get_optimizer(cost, global_step):
    if config.adam:
        # Don't use learning rate decay with Adam
        learning_rate = tf.constant(config.learning_rate, dtype=tf.float32)
        sgd = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam")
    else:
        # Set up Stochastic Gradient Descent Optimizer with exponential learning rate decay
        learning_rate = tf.train.exponential_decay(config.learning_rate, global_step=global_step,
                                                   decay_steps=100000, decay_rate=config.learning_decay_rate,
                                                   staircase=False, name="learning_rate")
        sgd = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="SGD")
    optimizer = sgd.minimize(cost, name='optimizer', global_step=global_step)
    return optimizer, learning_rate


def train(train_dataset, test_dataset):
    """
    Train the model

    :param train_dataset: Training dataset
    :param test_dataset: Testing/dev dataset
    :return: None
    """

    # Set up dataset iterators
    dataset_handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(dataset_handle,
                                                   train_dataset.output_types,
                                                   train_dataset.output_shapes)

    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    # MRI input and ground truth segmentations
    input, seg = iterator.get_next()

    # Create the model's computation graph and cost function
    output, is_training = UNet.model(input, seg)
    dice = dice_coeff(seg, output)
    cost = - dice

    # Define the optimization strategy
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer, learning_rate = _get_optimizer(cost, global_step)

    logger.info("Training...")
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        # Initialize graph and data iterators
        sess.run(init)
        train_handle = sess.run(train_iterator.string_handle())

        # Configure TensorBard
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('train_dice', dice)
        tf.summary.histogram("train_dice", dice)
        tf.summary.scalar('training_cost', cost)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir=tensorboard_dir)
        writer.add_graph(sess.graph) # Add the pretty graph viz

        # Training epochs
        for epoch in range(config.num_epochs):
            sess.run(train_iterator.initializer)

            epoch_cost = 0.0

            # Iterate through all batches in the epoch
            batch = 0
            while True:
                try:
                    _, c, d = sess.run([optimizer, cost, dice],
                                       feed_dict={is_training: True,
                                                  dataset_handle: train_handle})

                    logger.info("Epoch: %d, Batch %d: cost: %f, dice: %f" % (epoch, batch, c, d))
                    epoch_cost += epoch_cost / config.num_epochs
                    batch += 1

                    if batch % config.tensorboard_freq == 0:
                        logger.info("Logging TensorBoard data...")
                        # Write out stats for training
                        s = sess.run(merged_summary, feed_dict={is_training: False,
                                                                dataset_handle: train_dataset})
                        writer.add_summary(s, global_step=global_step)

                        # Generate stats for test dataset
                        sess.run(test_iterator.initializer)
                        test_handle = sess.run(test_iterator.string_handle())

                        test_dice_summary = tf.summary.histogram('test_dice', dice)
                        test_dice_avg_summary = tf.summary.scalar('test_dice_avg', tf.reduce_mean(dice))

                        test_dice, test_dice_summ, test_dice_avg_summ = \
                            sess.run([dice, test_dice_summary, test_dice_avg_summary],
                                     feed_dict={is_training: False,
                                                dataset_handle: test_handle})
                        
                        writer.add_summary(test_dice_summ, global_step=global_step)
                        writer.add_summary(test_dice_avg_summ, global_step=global_step)

                except tf.errors.OutOfRangeError:
                    logger.info("End of epoch %d" % epoch)
                    break
        logger.info("Training complete.")

        logger.info("Saving model to: %s ..." % config.model_file)
        saver = tf.train.Saver()
        saver.save(sess, config.model_file, global_step=global_step)
        logger.info("Model save complete.")


def parse_args():
    """
    Parse the command line options for this file
    :return: An argparse object containing parsed arguments
    """

    parser = argparse.ArgumentParser(description="Train the tumor segmentation model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    info_options = parser.add_argument_group("Info")
    info_options.add_argument("--job-dir", default=None, help="Job directory")
    info_options.add_argument("--job-name", default="segmentation", help="Job name")
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

    # Log to file if not on google cloud
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

    global config
    if args.config is not None:
        config = Configuration(config_file=args.config)
    else:
        config = Configuration()

    # Set the TensorBoard directory
    global tensorboard_dir
    tensorboard_dir = os.path.join(config.tensorboard_dir, _get_job_name())

    # Set random seed for reproducible results
    tf.set_random_seed(config.seed)

    logger.info("Creating data pre-processing pipeline...")
    logger.debug("BraTS data set directory: %s" % config.brats_directory)
    logger.debug("TFRecords: %s" % config.tfrecords_dir)

    train_dataset, test_dataset, validation_dataset = create_data_pipeline()

    logger.info("Initiating training...")
    logger.debug("TensorBoard Directory: %s" % config.tensorboard_dir)
    logger.debug("Model save file: %s" % config.model_file)
    logger.debug("Learning rate: %s" % config.learning_rate)
    logger.debug("Num epochs: %s" % config.num_epochs)
    logger.debug("Mini-batch size: %s" % config.mini_batch_size)
    train(train_dataset, test_dataset)

    logger.info("Exiting.")


if __name__ == "__main__":
    main()
