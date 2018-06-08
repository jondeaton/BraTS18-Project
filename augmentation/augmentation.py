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





# def random_flip(mri, seg):
#     axis = np.random.randint(3)
#     flipped_mri = np.flip(mri, axis=axis)
#     flipped_seg = np.flip(seg, axis=axis)
#     return flipped_mri, flipped_seg
#
# def add_noise(mri, seg):
#     new_mri = np.copy(mri)
#     for i in range(mri.shape[0]):
#         foreground = mri[i] != 0
#         std = np.std(mri[i][foreground])
#         noise = np.random.randn(*(mri.shape[1:])) * std * 0.3
#         new_mri[i][foreground] += noise[foreground]
#     return new_mri, seg
#
#
# def _blur(mri, seg):
#     blurred = gaussian_filter(mri, sigma=0.5)
#     return blurred, seg


def augment_training_set(train_dataset):
    """
    Augments a training data set

    :param train_dataset: tf.data.Dataset of training mris
    :return: Augmented data set
    """
    assert isinstance(train_dataset, tf.data.Dataset)

    with tf.variable_scope("augmentation"):

        # noisy = train_dataset.map(_add_noise)
        # train_dataset = train_dataset.concatenate(noisy)

        # todo: fix blurring augmentation
        # blurred = train_dataset.map(blur)
        # train_dataset.concatenate(blurred)

        flipped_lr = train_dataset.map(_flip_left_right)
        flipped_ud = train_dataset.map(_flip_up_down)
        flipped_fb = train_dataset.map(_flip_front_back)

        train_dataset = train_dataset\
            .concatenate(flipped_lr)\
            .concatenate(flipped_ud)\
            .concatenate(flipped_fb)

    return train_dataset


def _flip(mri, seg, axis):
    flipped_mri = tf.reverse(mri, axis=[axis])
    flipped_seg = tf.reverse(seg, axis=[axis])
    return flipped_mri, flipped_seg


def _flip_up_down(mri, seg):
    return _flip(mri, seg, 3)


def _flip_left_right(mri, seg):
    return _flip(mri, seg, 1)


def _flip_front_back(mri, seg):
    return _flip(mri, seg, 2)


def _add_noise(mri, seg):
    mean, std = tf.nn.moments(mri, axes=[0, 1, 2, 3])
    noise = tf.random_normal(shape=mri.shape, mean=0.0, stddev=std, dtype=tf.float32)
    non_zero = tf.not_equal(mri, tf.constant(0, dtype=tf.float32))
    _mri = tf.where(non_zero, mri + noise, mri)
    return _mri, seg
