# -*- coding: utf-8 -*-

from keras.models import Input, Model
from keras.layers import Conv3D, Conv2D, Concatenate, MaxPooling3D, MaxPooling2D, UpSampling3D, UpSampling2D, Dropout, \
    BatchNormalization, GlobalMaxPooling2D, GlobalMaxPooling3D, Dense
from settings import *

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
'''


def conv_block(m, dim, acti, bn, res, ndim, do=0):
    """
    Builds a convolution block.
    :param: Similar to paramaters in UNet(...)
    :return: A block of two times a convolution and batch normalization
    """
    n = Conv3D(dim, 3, activation=acti, padding='same')(m) if ndim == 3 else \
        Conv2D(dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    # print('Added dropout') if do else n
    n = Conv3D(dim, 3, activation=acti, padding='same')(n) if ndim == 3 else \
        Conv2D(dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
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


def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res, ndim, doeverylevel, al):
    """
    Builds one block in UNet. The function is recursive. The depth decreases with 1 every time.
    :param: Similar to paramaters in UNet(...)
    :return: A UNet of the depth specified in the input
    """
    o_aux = -1

    if depth > 0:
        n = conv_block(m, dim, acti, bn, res, ndim, do) if doeverylevel else \
            conv_block(m, dim, acti, bn, res, ndim)
        m = (MaxPooling3D((1, 2, 2))(n) if mp else Conv3D(dim, 3, strides=2, padding='same')(n)) if ndim == 3 else \
            (MaxPooling2D((2, 2))(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n))
        m, o_aux = level_block(m, int(inc * dim), depth - 1, inc, acti, do, bn, mp, up, res, ndim, doeverylevel, al)
        if up:
            m = UpSampling3D((1, 2, 2))(m) if ndim == 3 else \
                UpSampling2D((2, 2))(m)
            m = Conv3D(dim, 2, activation=acti, padding='same')(m) if ndim == 3 else \
                Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            raise Exception('Unet in 3D does not work without upsampling')
        n = Concatenate()([n, m])
        m = conv_block(n, dim, acti, bn, res, ndim, do) if doeverylevel else \
            conv_block(n, dim, acti, bn, res, ndim)
    else:
        m = conv_block(m, dim, acti, bn, res, ndim, do)
        if al:
            o_aux = aux_loss_block(m, ndim, inc, dim, acti)
    return m, o_aux


def UNet(img_shape, ndim, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False, doeverylevel=False, aux_loss=True):
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
                                ndim, doeverylevel, aux_loss)
    o_main = Conv3D(out_ch, 1, activation='sigmoid', name='main_output')(o_main) if ndim == 3 else \
             Conv2D(out_ch, 1, activation='sigmoid', name='main_output')(o_main)
    return Model(inputs=i, outputs=[o_main, o_aux]) if aux_loss else \
           Model(inputs=i, outputs=o_main)
