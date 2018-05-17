#!/usr/bin/env python
"""
File: model_UNet3D.py
Date: 5/15/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import tensorflow as tf


from keras import Input
from keras.models import Model
from keras.layers import Activation, Conv3D, MaxPooling3D
from keras.layers import BatchNormalization, Concatenate, UpSampling3D


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


def UNet3D(input_shape, filter_start=4, pool_size=(2, 2, 2)):
    """
    3D UNet Module implemented in Keras

    :param input_shape: The shape of the input layer
    :param filter_start: The number of filters to start with
    :param pool_size: The size of the pool
    :return: The model (not compiled)
    """

    X_input = Input(input_shape)

    # Unet applied to X_input

    # Level 1
    X = ConvBlockDown(X_input, num_filters=filter_start)
    X = ConvBlockDown(X, num_filters=filter_start)
    level_1 = X
    X = MaxPooling3D(pool_size=pool_size,
                     name='max_pool1')(X)

    # Level 2
    X = ConvBlockDown(X, num_filters=2 * filter_start)
    X = ConvBlockDown(X, num_filters=2 * filter_start)
    level_2 = X
    X = MaxPooling3D(pool_size=pool_size,
                     name='max_pool2')(X)

    # Level 3
    X = ConvBlockDown(X, num_filters=4 * filter_start)
    X = ConvBlockDown(X, num_filters=4 * filter_start)
    level_3 = X
    X = MaxPooling3D(pool_size=pool_size, name='max_pool3')(X)

    # Lowest Level
    X = ConvBlockDown(X, num_filters=4 * filter_start)
    X = ConvBlockDown(X, num_filters=4 * filter_start)

    # Up-convolutions
    X = ConvBlockUp(X, level_3, num_filters=4 * filter_start)
    X = ConvBlockUp(X, level_2, num_filters=2 * filter_start)
    X = ConvBlockUp(X, level_1, num_filters=filter_start)

    # Create model.
    model = Model(inputs=X_input, outputs=X, name='UNet')

    final_convolution = Conv3D(1,
                               (1, 1, 1),
                               data_format="channels_first",
                               strides=(1, 1, 1),
                               padding='same')(X)
    act = Activation("sigmoid")(final_convolution)
    model = Model(inputs=X_input, outputs=act)
    return model


def tf_down_block(input, is_training, num_filters, name='down_level'):
    """
    One level of

    :param input:
    :param is_training:
    :param num_filters:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        conv1 = tf_conv_block(input, is_training, num_filters=num_filters, name='conv1')
        conv2 = tf_conv_block(conv1, is_training, num_filters=num_filters, name='conv2')

        max_pool = tf.layers.max_pooling3d(conv2,
                                           pool_size=(2,2,2), strides=2,
                                           data_format='channels_first', name='max_pool')
        return max_pool, conv2


def tf_up_block(input, shortcut, is_training, num_filters, name="up_level"):
    with tf.variable_scope(name):
        deconv = tf.layers.conv3d_transpose(input,
                                            filters=num_filters, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                                            padding='same', data_format='channels_firat', name="deconv")

        tf.concat(values=[deconv, shortcut], axis=1, name="concat")
        conv1 = tf_conv_block(input, is_training, num_filters=num_filters, name='conv1')
        conv2 = tf_conv_block(conv1, is_training, num_filters=num_filters, name='conv2')
        return conv2

def tf_conv_block(input, is_training, num_filters, name='conv'):
    """
    Convolution and batch normalization layer

    :param input: The input tensor
    :param is_training: Boolean tensor whether it is beign run on training or not
    :param num_filters: The number of filters to convolve on the input
    :param name: Name of the convolutional block
    :return: Tensor after convolution and batch normalization
    """
    with tf.variable_scope(name):
        kernel_initializer = tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
        bias_initializer = tf.zeros_initializer(dtype=tf.float32)

        conv = tf.layers.conv3d(input,
                                filters=num_filters, kernel_size=(3,3,3), strides=(1,1,1), padding='same',
                                data_format='channels_first', activation=None, use_bias=True,
                                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

        # Batch normalization before activation
        bn = tf.layers.batch_normalization(conv,
                                           axis=-1, momentum=0.9,
                                           epsilon=0.001, center=True, scale=True,
                                           training=is_training, name='bn')

        # Activation after batch normalization
        act = tf.nn.relu(bn)
        tf.summary.histogram('activations', act)
        tf.summary.scalar('sparsity', tf.nn.zero_fraction(act))
        return act



def tf_dice_coefficient_loss(seg_true, seg_pred):
    with tf.variable_scope("dice_coff_loss"):
        seg_true_flat = tf.layers.flatten(seg_true)
        seg_pred_flat = tf.layers.flatten(seg_pred)
        intersection = tf.multiply(seg_true_flat, seg_pred_flat, name="intersection")
        intersect = tf.reduce_sum(intersection)

        smooth = 0.01
        dice = tf.divide(2 * intersect + smooth, tf.reduce_sum(seg_true_flat) + tf.reduce_sum(seg_pred_flat) + smooth)
        return dice

def UNet3D_tf(input_shape, output_shape):

    input = tf.placeholder(dtype=tf.float32, shape=input_shape)
    segs = tf.placeholder(dtype=tf.float32, shape=output_shape)
    is_training = tf.placeholder(tf.bool)

    with tf.variable_scope("down"):
        level1, l1_conv = tf_down_block(input,  is_training, num_filters=8,  name="down_level1")
        level2, l2_conv = tf_down_block(level1, is_training, num_filters=16, name="down_level2")
        level3, l3_conv = tf_down_block(level2, is_training, num_filters=32, name="down_level3")

    with tf.variable_scope("level4"):
        conv1 = tf_conv_block(level3, is_training, num_filters=32, name="conv1")
        conv2 = tf_conv_block(conv1,  is_training, num_filters=32, name="conv2")

    with tf.variable_scope("up"):
        level3_up = tf_up_block(conv2,     l3_conv, is_training, num_filters=16, name="level3")
        level2_up = tf_up_block(level3_up, l2_conv, is_training, num_filters=8, name="level2")
        level1_up = tf_up_block(level2_up, l1_conv, is_training, num_filters=4, name="level3")

    with tf.variable_scope("output"):
        kernel_initializer = tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
        bias_initializer = tf.zeros_initializer(dtype=tf.float32)
        output = tf.layers.conv3d(level1_up,
                                      filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                      padding='same',
                                      data_format='channels_first', activation='sigmoid', use_bias=True,
                                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

        tf.summary.histogram('activations', output)
        tf.summary.scalar('sparsity', tf.nn.zero_fraction(output))

    cost = tf_dice_coefficient_loss(segs, output)


    return input, segs, cost, is_training


def train():

    input, segs, cost, is_training = UNet3D_tf(mri_shape, seg_shape)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    sgd = tf.train.AdamOptimizer(learning_rate=learning_rate, name="gradient-descent")
    optimizer = sgd.minimize(cost, name='optimizer', global_step=global_step)
