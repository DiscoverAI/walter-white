import logging
import os
from functools import partial

import tensorflow as tf

LOG = logging.getLogger(__name__)


def parse_line(line):
    split_line = tf.strings.split(line, sep=',')
    return tf.strings.to_number(split_line, out_type=tf.dtypes.double)


def load_dataset(dataset_path):
    absolute_dataset_path = os.path.realpath(dataset_path)
    raw_lines_dataset = tf.data.TextLineDataset(tf.data.Dataset.list_files(absolute_dataset_path, shuffle=False))
    return raw_lines_dataset.map(partial(parse_line))
