# Import tensorflow and numpy
import tensorflow as tf
import numpy as np
import math as m

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
        ## Perform lifting convolution
        # The kernels used in the lifting layer
        print(tensor_in.get_shape().as_list())
        print('kernel: {}'.format([self.Nxy, self.Nxy, Nc_in, Nc_out]))

        with tf.variable_scope("Layer_{}".format(self.next_layer())):
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

            Nc_in = Nc_out

            tensor_in = tensor_out

        print(tensor_in.get_shape().as_list())

        with tf.variable_scope("Layer_{}".format(self.next_layer())):
            ## Perform group convolution
            # The kernels used in the group convolution layer
            kernels_raw = tf.get_variable(
                'kernel',
                [self.Nxy, self.Nxy, self.Ntheta, Nc_in, Nc_out],
                initializer=self.weight_initializer(self.Nxy * self.Nxy * self.Ntheta * Nc_in, Nc_out)
            )
            tf.add_to_collection('raw_kernels', kernels_raw)
            bias = tf.get_variable(  # Same bias for all orientations
                "bias",
                [1, 1, 1, 1, Nc_out],
                initializer=tf.constant_initializer(value=0.01))
            # The group convolution layer
            tensor_out, kernels_formatted = se2cnn.layers.se2n_se2n(
                input_tensor=tensor_in,
                kernel=kernels_raw
            )
            tensor_out = tensor_out + bias

            ## Perform max-pooling
            tensor_out = se2cnn.layers.spatial_max_pool(input_tensor=tensor_out, nbOrientations=self.Ntheta)

            ## Apply ReLU
            tensor_out = tf.nn.relu(tensor_out)
        print(tensor_out.get_shape().as_list())
        return tensor_out

    def conc(self, tensor_in):
        # Concatenate the orientation and channel dimension
        tensor_out = tf.concat([tensor_in[:, :, :, i, :] for i in range(self.Ntheta)], 3)
        # Nc_in = tensor_in.get_shape().as_list()[-1]
        return tensor_out

    def max_pool(self, tensor_in):
        ## Perform (spatial) max-pooling
        tensor_out = se2cnn.layers.spatial_max_pool(input_tensor=tensor_in, nbOrientations=self.Ntheta)
        return tensor_out

    def concatenate(self, n, m):
        return tf.concat([n, m], n.get_shape().as_list()[-1])

    def level_block(self, m, depth, Nc_in, Nc_out):
        if depth > 0:
            n = self.conv_block(m, Nc_in, Nc_out)
            print('here')
            m = self.max_pool(n)
            m = self.conc(m)
            m = self.level_block(m, depth - 1, Nc_out, Nc_out * self.inc)
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
        self.Nc = 4  # Number of channels in the initial layer

        self.inc = 4  # Incremental rate of feature maps

        inputs_ph = tf.placeholder(dtype=tf.float32, shape=[None] + list(img_shape))
        outputs_ph = tf.placeholder(dtype=tf.int16, shape=[None] + list(img_shape))

        m = self.level_block(inputs_ph, depth, 1, 4)





if __name__ == '__main__':
    UNet((480, 480, 1), depth=3)
