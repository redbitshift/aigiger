"""A.I. Giger training."""

from os import path

import numpy as np
import tensorflow as tf

from model import make_discriminator, make_generator

class Giger:
    FAKE_POOL_SIZE = 100
    MAX_TRUTH = 100

    def __init__(self, config):
        self.config = config

        dims = [config.batch_size, config.size, config.size, 3]
        self.noise = tf.placeholder(tf.float32, dims, name='noise')
        self.maybe = tf.placeholder(tf.float32, dims, name='maybe')

        with tf.variable_scope('model'):
            self.d = make_discriminator(self.maybe)
            self.g = make_generator(self.noise)

    def read_truth_data(self):
        conf = self.config
        image = self.make_image_pipeline(conf.truth_dir)

        with tf.Session() as session:
            init = (tf.global_variables_initializer(),
                    tf.local_variables_initializer())
            session.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            dims = (self.MAX_TRUTH, conf.batch_size, conf.size, conf.size, 3)
            self.truth = np.zeros(dims)

            for i in range(Giger.MAX_TRUTH):
                self.truth[i] = session.run(image)

            coord.request_stop()
            coord.join(threads)

    def make_image_pipeline(self, image_dir):
        pattern = path.join('.', image_dir, '*.jpeg')
        filenames = tf.train.match_filenames_once(pattern)
        truth_queue = tf.train.string_input_producer(filenames)
        reader = tf.WholeFileReader()
        _, image_bytes = reader.read(truth_queue)
        image = tf.image.decode_jpeg(image_bytes, 3)
        return tf.to_float(image) / 127.5 - 1.
