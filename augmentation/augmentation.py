#!/usr/bin/env python
"""
File: augmentation
Date: 5/9/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import random
import itertools

import numpy as np
import tensorflow as tf

from nilearn.image import new_img_like, resample_to_img
from scipy.ndimage.filters import gaussian_filter


from keras.preprocessing.image import ImageDataGenerator



def random_flip(mri, seg):
    axis = np.random.randint(3)
    flipped_mri = np.flip(mri, axis=axis)
    flipped_seg = np.flip(seg, axis=axis)
    return flipped_mri, flipped_seg

def add_noise(mri, seg):
    new_mri = np.copy(mri)
    for i in range(mri.shape[0]):
        foreground = mri[i] != 0
        std = np.std(mri[i][foreground])
        noise = np.random.randn(*(mri.shape[1:])) * std * 0.3
        new_mri[i][foreground] += noise[foreground]
    return new_mri, seg

def blur(mri, seg):
    blurred = gaussian_filter(mri, sigma=0.5)
    return blurred, seg


def _random_flip(mri, seg):
    axis = tf.random_uniform((1,), minval=1, maxval=3, dtype=tf.int32)
    flipped_mri = tf.reverse(mri, axis=axis)
    flipped_seg = tf.reverse(seg, axis=axis)
    return flipped_mri, flipped_seg


def _add_noise(mri, seg):
    zero = tf.constant(0, dtype=tf.float32)
    for i in range(tf.shape(mri)[0]):
        img = mri[0]
        non_zero = tf.not_equal(img, zero)

        std = tf.nn.moments(mri[0], axes=[0, 1, 2])
        noise = tf.random_normal(shape=tf.shape(mri), mean=0.0, stddev=std, dtype=tf.float32)
        mri[i] = tf.where(non_zero, img + noise, img)

    return mri, seg


def augment_training_set(train_dataset):
    """
    Augments a training data set

    :param train_dataset: tf.data.Dataset of training mris
    :return: Augmented data set
    """
    assert isinstance(train_dataset, tf.data.Dataset)

    with tf.variable_scope("augmentation"):
        flipped = train_dataset.map(_random_flip)
        train_dataset = train_dataset.concatenate(flipped)

        # todo: fix
        # noisy = train_dataset.map(add_noise)
        # train_dataset = train_dataset.concatenate(noisy)

        # todo: fix blurring augmentation
        # blurred = train_dataset.map(blur)
        # train_dataset.concatenate(blurred)

    return train_dataset
