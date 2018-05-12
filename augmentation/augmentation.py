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

def random_flip_tf(mri, seg):
    axis = np.random.randint(3)
    flipped_mri = tf.reverse(mri, axis=axis)
    flipped_seg = tf.reverse(seg, axis=axis)
    return flipped_mri, flipped_seg


def random_scale(mri, seg):
    # todo: yeah this is broken
    scale_factor = np.random.normal(1, 0.25, 3)
    image = new_img_like()
    new_affine = np.copy(image.affine)
    new_affine[:3, :3] = image.affine[:3, :3] * scale_factor
    new_affine[:, 3][:3] = image.affine[:, 3][:3] + (image.shape * np.diag(image.affine)[:3] * (1 - scale_factor)) / 2
    return new_img_like(image, data=image.get_data(), affine=new_affine)


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
    return blurred


def augment_training_set(train_dataset):
    """
    Augments a training data set

    :param train_dataset: tf.data.Dataset of training mris
    :return: Augmented data set
    """
    assert isinstance(train_dataset, tf.data.Dataset)

    flipped = train_dataset.map(random_flip_tf)
    train_dataset = train_dataset.concatenate(flipped)

    # scaled = train_dataset.map(random_scale)
    # train_dataset = train_dataset.concatenate(scaled)

    noisy = train_dataset.map(add_noise)
    train_dataset = train_dataset.concatenate(noisy)

    blurred = train_dataset.map(blur)
    train_dataset.concatenate(blurred)

    return train_dataset.shuffle()
