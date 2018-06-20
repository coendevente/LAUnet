# Import tensorflow and numpy
import tensorflow as tf
import numpy as np
import math as m
from keras.layers import UpSampling2D

# Import the library
import se2cnn.layers


class UNet:
    # Xavier's/He-Rang-Zhen-Sun initialization for layers that are followed ReLU
    def weight_initializer(self, n_in, n_out):
        return tf.random_normal_initializer(mean=0.0, stddev=m.sqrt(2.0 / (n_in))
        )

    def size_of(self, tensor):
        # Multiply elements one by one
        result = 1
        for x in tensor.get_shape().as_list():
             result = result * x
        return result

    def next_layer(self):
        l = self.layer_nr
        self.layer_nr += 1
        return l

    def conv_block(self, tensor_in, Nc_in, Nc_out):
        with tf.variable_scope("Layer_{}".format(self.next_layer())) as _scope:
            # ## Settings
            # Nc_out = self.Nc

            ## Perform lifting convolution
            # The kernels used in the lifting layer
            kernels_raw = tf.get_variable(
                'kernel',
                [self.Nxy, self.Nxy, Nc_in, Nc_out],
                initializer=self.weight_initializer(self.Nxy * self.Nxy * Nc_in, Nc_out))
            tf.add_to_collection('raw_kernels', kernels_raw)
            bias = tf.get_variable(  # Same bias for all orientations
                "bias",
                [1, 1, 1, 1, Nc_out],
                initializer=tf.constant_initializer(value=0.01))
            # Lifting layer
            tensor_out, kernels_formatted = se2cnn.layers.z2_se2n(
                input_tensor=tensor_in,
                kernel=kernels_raw,
                orientations_nb=self.Ntheta)
            # Add bias
            tensor_out = tensor_out + bias

            ## Apply ReLU
            tensor_out = tf.nn.relu(tensor_out)

            ## Prepare for the next layer
            tensor_in = tensor_out
            Nc_in = Nc_out

        with tf.variable_scope("Layer_{}".format(self.next_layer())) as _scope:
            # ## Settings
            # Nc_out = 2 * Nc

            ## Perform group convolution
            # The kernels used in the group convolution layer
            kernels_raw = tf.get_variable(
                'kernel',
                [self.Nxy, self.Nxy, self.Ntheta, Nc_in, Nc_out],
                initializer=self.weight_initializer(self.Nxy * self.Nxy * self.Ntheta * Nc_in, Nc_out))
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
        tensor_in = tf.concat([tensor_in[:, :, :, i, :] for i in range(self.Ntheta)], 3)
        Nc_in = tensor_in.get_shape().as_list()[-1]

        # 2D convolution layer
        with tf.variable_scope("Layer_{}".format(self.next_layer())) as _scope:
            # ## Settings
            # Nc_out = 4 * Nc

            ## Perform group convolution
            # The kernels used in the group convolution layer
            kernels_raw = tf.get_variable(
                'kernel',
                [self.Nxy, self.Nxy, Nc_in, Nc_out],
                initializer=self.weight_initializer(self.Nxy * self.Nxy * Nc_in, Nc_out))
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
                padding="VALID")
            tensor_out = tensor_out + bias

            ## Apply ReLU
            tensor_out = tf.nn.relu(tensor_out)

            ## Prepare for the next layer
            tensor_in = tensor_out
            Nc_in = Nc_out

        print(tensor_in.get_shape())

        return tensor_out

    # def conc(self, tensor_in):
    #
    #     return tensor_out

    def max_pool(self, tensor_in):
        tensor_out = tf.nn.max_pool(
            tensor_in,
            [0, 2, 2, 0],
            [1, 2, 2, 1],
            'SAME',
            # data_format='NHWC',
            # name=None
        )
        return tensor_out

    def concatenate(self, n, m):
        print('n.get_shape() == {}'.format(n.get_shape()))
        print('m.get_shape() == {}'.format(m.get_shape()))
        return tf.concat([n, m], n.get_shape().as_list()[-1])

    def up_conv(self, m):
        # sz = m.get_shape() * 2
        # print(sz)

        # tensor_out = tf.image.resize_images(
        #     m,
        #     [sz[1]*2, sz[2]*2],
        #     # method=ResizeMethod.BILINEAR,
        #     # align_corners=False
        # )
        tensor_out = UpSampling2D((2, 2))(m)

        tensor_out = self.conv_block(tensor_out, m.get_shape[-1], m.get_shape[-1]/2)

        return tensor_out

    def level_block(self, m, depth, Nc_in, Nc_out):
        if depth > 0:
            n = self.conv_block(m, Nc_in, Nc_out)
            print('here')
            m = self.max_pool(n)
            m = self.level_block(m, depth - 1, Nc_out, Nc_out * self.inc)
            m = self.up_conv(m)
            n = self.concatenate(n, m)
            m = self.conv_block(n, Nc_out, Nc_in)
        else:
            m = self.conv_block(m, Nc_in, Nc_out)
        return m

    def __init__(self, img_shape, depth=3):
        graph = tf.Graph()
        graph.as_default()
        tf.reset_default_graph()
        self.layer_nr = 0

        self.Ntheta = 12  # Kernel size in angular direction
        self.Nxy = 3  # Kernel size in spatial direction
        self.Nc = 64  # Number of channels in the initial layer

        self.inc = 2  # Incremental rate of feature maps

        inputs_ph = tf.placeholder(dtype=tf.float32, shape=[None] + list(img_shape))
        outputs_ph = tf.placeholder(dtype=tf.int16, shape=[None] + list(img_shape))

        m = self.level_block(inputs_ph, depth, 1, self.Nc)


if __name__ == '__main__':
    UNet((480, 480, 1), depth=3)
