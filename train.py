"""A.I. Giger training."""

import sys

import gflags
import tensorflow as tf

from giger import Giger

gflags.DEFINE_integer('batch_size', 1, 'Batch size')
gflags.DEFINE_integer('size', 128, 'Image size')
gflags.DEFINE_string('truth_dir', 'truth', 'Truth data directory')

F = gflags.FLAGS


def main(argv):
    giger = Giger(F)
    giger.read_truth_data()


if __name__ == '__main__':
    F(sys.argv)
    sys.exit(main(sys.argv))
