#!/usr/bin/env python
"""
File: UNet3D.py
Date: 5/15/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import tensorflow as tf


def down_block(input, is_training, num_filters, name='down_level'):
    with tf.variable_scope(name):
        conv1 = conv_block(input, is_training, num_filters=num_filters, name='conv1')
        conv2 = conv_block(conv1, is_training, num_filters=num_filters, name='conv2')

        max_pool = tf.layers.max_pooling3d(conv2,
                                           pool_size=(2,2,2), strides=2,
                                           data_format='channels_first', name='max_pool')
        return max_pool, conv2


def up_block(input, shortcut, is_training, num_filters, name="up_level"):
    with tf.variable_scope(name):
        deconv = tf.layers.conv3d_transpose(input,
                                            filters=num_filters, kernel_size=(3,3,3), strides=(2,2,2),
                                            padding='same', data_format='channels_first', name="deconv")

        concat = tf.concat(values=[deconv, shortcut], axis=1, name="concat")
        conv1 = conv_block(concat, is_training, num_filters=num_filters, name='conv1')
        conv2 = conv_block(conv1, is_training, num_filters=num_filters, name='conv2')
        return conv2


def conv_block(input, is_training, num_filters, name='conv'):
    """
    Convolution and batch normalization layer

    :param input: The input tensor
    :param is_training: Boolean tensor whether it is being run on training or not
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
        act = tf.nn.relu(bn, name="bn-relu")
        tf.summary.histogram('activations', act)
        tf.summary.scalar('sparsity', tf.nn.zero_fraction(act))
        return act


def model(input, seg, multi_class):

    is_training = tf.placeholder(tf.bool)

    with tf.variable_scope("down"):
        level1, l1_conv = down_block(input, is_training, num_filters=8, name="down_level1")
        level2, l2_conv = down_block(level1, is_training, num_filters=16, name="down_level2")
        level3, l3_conv = down_block(level2, is_training, num_filters=32, name="down_level3")

    with tf.variable_scope("level4"):
        conv1 = conv_block(level3, is_training, num_filters=32, name="conv1")
        conv2 = conv_block(conv1, is_training, num_filters=32, name="conv2")

    with tf.variable_scope("up"):
        level3_up = up_block(conv2, l3_conv, is_training, num_filters=16, name="level3")
        level2_up = up_block(level3_up, l2_conv, is_training, num_filters=8, name="level2")
        level1_up = up_block(level2_up, l1_conv, is_training, num_filters=4, name="level1")

    with tf.variable_scope("output"):
        kernel_initializer = tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
        bias_initializer = tf.zeros_initializer(dtype=tf.float32)

        if multi_class:
            final_conv = tf.layers.conv3d(level1_up,
                                      filters=4, kernel_size=(1,1,1), strides=(1,1,1), padding='same',
                                      data_fomat='channels_first', activation=None, use_bias=True,
                                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
            output = tf.nn.softmax(final_conv, axis=1, name="softmax")
        else:
            output = tf.layers.conv3d(level1_up,
                                          filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                          data_format='channels_first', activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

        tf.summary.histogram('activations', output)
        tf.summary.scalar('sparsity', tf.nn.zero_fraction(output))

    return output, is_training
