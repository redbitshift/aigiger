"""A.I. Giger model."""

import tensorflow as tf

def make_discriminator(candidate):
    k = 4
    s0, s1 = 2, 1
    n_filters = 64

    with tf.variable_scope('discriminator'):
       layer = convolve(candidate, n_filters, k, s0, 'c1', 'same', norm=False)
       layer = convolve(layer, n_filters * 2, k, s0, 'c2', 'same')
       layer = convolve(layer, n_filters * 4, k, s0, 'c3', 'same')
       layer = convolve(layer, n_filters * 8, k, s1, 'c4', 'same')
       layer = convolve(layer, 1, k, s1, 'c5', 'same', norm=False, relu=False)

       return layer


def make_generator(input_layer):
    k0, k1 = 7, 3  # Kernel sizes.
    s0, s1 = 1, 2  # Strides.
    n_filters = 32

    with tf.variable_scope('generator'):
        layer = pad(input_layer, k1)
        layer = convolve(layer, n_filters, k0, s0, 'c1', 'valid')
        layer = convolve(layer, n_filters * 2, k1, s1, 'c2', 'same')
        layer = convolve(layer, n_filters * 4, k1, s1, 'c3', 'same')

        layer = make_resnet_blocks(layer, n_filters * 4, 6)

        layer = deconvolve(layer, n_filters * 2, k1, s1, 'c4')
        layer = deconvolve(layer, n_filters, k1, s1, 'c5')
        layer = pad(layer, k1)
        layer = convolve(layer, 3, k0, s0, 'c6', 'same', False)

        layer = tf.nn.tanh(layer)

        return layer


def make_resnet_blocks(layer, n_filters, n_blocks):
    for i in range(n_blocks):
        layer = make_resnet_block(layer, n_filters, 'res%d' % i)
    return layer


def make_resnet_block(input_layer, n_filters, name):
    def conv(layer, name, relu=True):
        return convolve(layer, n_filters, 3, 1, name, 'valid', relu)

    with tf.variable_scope(name):
        layer = pad(input_layer, 1)
        layer = conv(layer, 'c1')
        layer = pad(layer, 1)
        layer = conv(layer, 'c2', False)
        return tf.nn.leaky_relu(layer + input_layer)


def pad(layer, pad):
    padding = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
    return tf.pad(layer, padding, 'REFLECT')


def convolve(layer, n_filters, size, stride, name, 
             padding='valid', norm=True, relu=True):
    with tf.variable_scope(name):
        layer = tf.layers.conv2d(
            layer, n_filters, size, stride, padding=padding,
            activation=tf.nn.leaky_relu if relu else None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.2))

        # Note: this applies norm after relu which differs from the orig paper.
        if norm:
            layer = normalize(layer)

        return layer


def deconvolve(layer, n_filters, size, stride, name, norm=True):
    with tf.variable_scope(name):
        layer = tf.layers.conv2d_transpose(
            layer, n_filters, (size, size), (stride, stride), 'same', 
            kernel_initializer=tf.truncated_normal_initializer(stddev=.2),
            activation=tf.nn.leaky_relu)

        # Note: this applies norm after relu which differs from the orig paper.
        if norm:
            layer = normalize(layer)

        return layer


def normalize(layer):
    with tf.variable_scope('instance_norm'):
        epsilon = 1e-5
        mean, var = tf.nn.moments(layer, [1, 2], keep_dims=True)
        n_filters = layer.get_shape()[-1]
        scale = tf.get_variable(
            'scale', [n_filters],
            initializer=tf.truncated_normal_initializer(mean=1., stddev=.02))
        offset = tf.get_variable(
            'offset', [n_filters], initializer=tf.constant_initializer(0.))

        return scale * tf.div(layer - mean, tf.sqrt(var + epsilon)) + offset
