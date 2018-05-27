#!/usr/bin/env python
"""
File: keras_UNet3D
Date: 5/26/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""



from keras import Input
from keras.models import Model
from keras.layers import Activation, Conv3D, MaxPooling3D
from keras.layers import BatchNormalization, Concatenate, UpSampling3D
from keras.initializers import glorot_uniform, glorot_normal

def ConvBlockDown(input_layer, num_filters=32):
    strides = (1, 1, 1)
    kernel = (3, 3, 3)

    layer = Conv3D(num_filters,
                   kernel,
                   data_format="channels_first",
                   strides=strides,
                   kernel_initializer=glorot_normal,
                   bias_initializer=glorot_uniform,
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
               kernel_initializer=glorot_normal,
               bias_initializer=glorot_uniform,
               padding='same')(X)

    X = Conv3D(num_filters,
               kernel,
               data_format="channels_first",
               strides=strides,
               kernel_initializer=glorot_normal,
               bias_initializer=glorot_uniform,
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
                               kernel_initializer=glorot_normal,
                               bias_initializer=glorot_uniform,
                               padding='same')(X)
    act = Activation("sigmoid")(final_convolution)
    model = Model(inputs=X_input, outputs=act)
    return model
