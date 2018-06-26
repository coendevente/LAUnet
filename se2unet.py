# -*- coding: utf-8 -*-

from keras.models import Input, Model
from keras.layers import Conv3D, Conv2D, Concatenate, MaxPooling3D, MaxPooling2D, UpSampling3D, UpSampling2D, Dropout, \
    BatchNormalization, GlobalMaxPooling2D, GlobalMaxPooling3D, Dense, Lambda
from settings import *

# Import the library
import se2cnn.layers
import tensorflow as tf
import math as m

'''
U-Net: Convolutional Networks for Biomedical Image Segmentation
(https://arxiv.org/abs/1505.04597)
---
img_shape: (height, width, channels)
out_ch: number of output channels
start_ch: number of channels of the first conv
depth: zero indexed depth of the U-structure
inc_rate: rate at which the conv channels will increase
activation: activation function after convolutions
dropout: amount of dropout in the contracting part
batchnorm: adds Batch Normalization if true
maxpool: use strided conv instead of maxpooling if false
upconv: use transposed conv instead of upsamping + conv if false
residual: add residual connections around each conv block if true

https://github.com/pietz/unet-keras/blob/master/unet.py
'''


# Xavier's/He-Rang-Zhen-Sun initialization for layers that are followed ReLU
def weight_initializer(n_in, n_out):
    return tf.random_normal_initializer(mean=0.0, stddev=m.sqrt(2.0 / (n_in))
    )


def size_of(tensor):
    # Multiply elements one by one
    result = 1
    for x in tensor.get_shape().as_list():
         result = result * x
    return result


global layer_nr
layer_nr = 0


def next_layer():
    global layer_nr
    l = layer_nr
    layer_nr += 1
    return l


def se2conv(tensor_in, Nxy, Nc_out, Ntheta):
    Nc_in = tensor_in.get_shape().as_list()[-1]

    with tf.variable_scope("Layer_{}".format(next_layer())) as _scope:
        # ## Settings
        # Nc_out = Nc

        ## Perform lifting convolution
        # The kernels used in the lifting layer
        kernels_raw = tf.get_variable(
            'kernel',
            [Nxy, Nxy, Nc_in, Nc_out],
            initializer=weight_initializer(Nxy * Nxy * Nc_in, Nc_out))
        tf.add_to_collection('raw_kernels', kernels_raw)
        bias = tf.get_variable(  # Same bias for all orientations
            "bias",
            [1, 1, 1, 1, Nc_out],
            initializer=tf.constant_initializer(value=0.01))
        # Lifting layer
        tensor_out, kernels_formatted = se2cnn.layers.z2_se2n(
            input_tensor=tensor_in,
            kernel=kernels_raw,
            orientations_nb=Ntheta)
        # Add bias
        tensor_out = tensor_out + bias

        ## Apply ReLU
        tensor_out = tf.nn.relu(tensor_out)

        ## Prepare for the next layer
        tensor_in = tensor_out
        Nc_in = Nc_out

    with tf.variable_scope("Layer_{}".format(next_layer())) as _scope:
        # ## Settings
        # Nc_out = 2 * Nc

        ## Perform group convolution
        # The kernels used in the group convolution layer
        kernels_raw = tf.get_variable(
            'kernel',
            [Nxy, Nxy, Ntheta, Nc_in, Nc_out],
            initializer=weight_initializer(Nxy * Nxy * Ntheta * Nc_in, Nc_out))
        tf.add_to_collection('raw_kernels', kernels_raw)
        bias = tf.get_variable(  # Same bias for all orientations
            "bias",
            [1, 1, 1, 1, Nc_out],
            initializer=tf.constant_initializer(value=0.01))
        # The group convolution layer
        tensor_out, kernels_formatted = se2cnn.layers.se2n_se2n(
            input_tensor=tensor_in,
            kernel=kernels_raw)
        tensor_out = tensor_out + bias

        ## Apply ReLU
        tensor_out = tf.nn.relu(tensor_out)

        ## Prepare for the next layer
        tensor_in = tensor_out
        Nc_in = Nc_out

    # Concatenate the orientation and channel dimension
    tensor_in = tf.concat([tensor_in[:, :, :, i, :] for i in range(Ntheta)], 3)
    Nc_in = tensor_in.get_shape().as_list()[-1]

    # 2D convolution layer
    with tf.variable_scope("Layer_{}".format(next_layer())) as _scope:
        # ## Settings
        # Nc_out = 4 * Nc

        ## Perform group convolution
        # The kernels used in the group convolution layer
        kernels_raw = tf.get_variable(
            'kernel',
            [Nxy, Nxy, Nc_in, Nc_out],
            initializer=weight_initializer(Nxy * Nxy * Nc_in, Nc_out))
        tf.add_to_collection('raw_kernels', kernels_raw)
        bias = tf.get_variable(  # Same bias for all orientations
            "bias",
            [1, 1, 1, Nc_out],
            initializer=tf.constant_initializer(value=0.01))
        # Convolution layer
        tensor_out = tf.nn.conv2d(
            input=tensor_in,
            filter=kernels_raw,
            strides=[1, 1, 1, 1],
            padding="SAME")
        tensor_out = tensor_out + bias

        ## Apply ReLU
        tensor_out = tf.nn.relu(tensor_out)

        ## Prepare for the next layer
        tensor_in = tensor_out
        Nc_in = Nc_out

    print(tensor_in.get_shape())

    return tensor_out


def conv_block(m, dim, acti, bn, res, ndim, nr_conv_per_block, n_theta, do=0):
    """
    Builds a convolution block.
    :param: Similar to paramaters in UNet(...)
    :return: A block of two times a convolution and batch normalization
    """

    n = m

    for i in range(nr_conv_per_block):
        n = Lambda(lambda x: se2conv(x, 3, dim, n_theta))(n)
        n = BatchNormalization()(n) if bn else n
        n = Dropout(do)(n) if do and i == 0 else n

    return Concatenate()([m, n]) if res else n


def aux_loss_block(m, ndim, inc, dim, acti):
    if ndim == 2:
        o = MaxPooling2D((2, 2))(m)
        o = Conv2D(int(dim / (inc ** 2)), 3, activation=acti, padding='same')(o)
        o = Conv2D(int(dim / (inc ** 4)), 3, activation=acti, padding='same')(o)
        o = GlobalMaxPooling2D()(o)
        o = Dense(1, activation='sigmoid', name='aux_output')(o)
    else:
        raise Exception('No global max pooling in 3D')
    return o


def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res, ndim, doeverylevel, al, nr_conv_per_block, n_theta):
    """
    Builds one block in UNet. The function is recursive. The depth decreases with 1 every time.
    :param: Similar to paramaters in UNet(...)
    :return: A UNet of the depth specified in the input
    """
    o_aux = -1

    if depth > 0:
        n = conv_block(m, dim, acti, bn, res, ndim, nr_conv_per_block, n_theta, do) if doeverylevel else \
            conv_block(m, dim, acti, bn, res, ndim, nr_conv_per_block, n_theta)
        m = (MaxPooling3D((1, 2, 2))(n) if mp else Conv3D(dim, 3, strides=2, padding='same')(n)) if ndim == 3 else \
            (MaxPooling2D((2, 2))(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n))
        m, o_aux = level_block(m, int(inc * dim), depth - 1, inc, acti, do, bn, mp, up, res, ndim, doeverylevel, al,
                               nr_conv_per_block, n_theta)
        if up:
            m = UpSampling3D((1, 2, 2))(m) if ndim == 3 else \
                UpSampling2D((2, 2))(m)
            m = Conv3D(dim, 2, activation=acti, padding='same')(m) if ndim == 3 else \
                Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            raise Exception('Unet in 3D does not work without upsampling')
        n = Concatenate()([n, m])
        m = conv_block(n, dim, acti, bn, res, ndim, nr_conv_per_block, n_theta, do) if doeverylevel else \
            conv_block(n, dim, acti, bn, res, ndim, nr_conv_per_block, n_theta)
    else:
        m = conv_block(m, dim, acti, bn, res, ndim, nr_conv_per_block, n_theta, do)
        if al:
            o_aux = aux_loss_block(m, ndim, inc, dim, acti)
    return m, o_aux


def UNet(img_shape, ndim, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False, doeverylevel=False, aux_loss=False,
         nr_conv_per_block=2, n_theta=4):
    """
    Makes UNet model.

    :param img_shape: Input shape
    :param out_ch: Number of output channels
    :param start_ch: Number of feature maps in output of first convolution
    :param depth: Number of concatenations in UNet
    :param inc_rate: Rate with which the number of feature maps increases as the depth increases
    :param activation: Activation method of convolutions (except for final convolution)
    :param dropout: Dropout fraction at bottom of UNet
    :param batchnorm: True iff Batch Normalization is applied
    :param maxpool: True iff Max Pooling is applied when leveling down in the UNet. False iff when leveling down in the
    UNet
    :param upconv: True iff going up in a level is done by upsampling and convolution. False iff
    :param residual: True iff concatenation needs to be performed when upsampling. Otherwise only feature maps from
    upsampling are used
    :return: UNet as Keras object
    """
    i = Input(shape=img_shape)
    o_main, o_aux = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual,
                                ndim, doeverylevel, aux_loss, nr_conv_per_block, n_theta)
    o_main = Conv2D(out_ch, 1, activation='sigmoid', name='main_output')(o_main)
    return Model(inputs=i, outputs=[o_main, o_aux]) if aux_loss else \
           Model(inputs=i, outputs=o_main)

if __name__ == '__main__':
    UNet((480, 480, 1), 2, depth=3)