import logging
import os
from functools import partial

import tensorflow as tf

LOG = logging.getLogger(__name__)


def parse_line(dictionary_size, line):
    split_line = tf.strings.split(line, sep=',')
    return tf.divide(tf.strings.to_number(split_line, out_type=tf.dtypes.double), dictionary_size)


def load_dataset(dataset_path, dictionary_size):
    absolute_dataset_path = os.path.realpath(dataset_path)
    all_files = tf.data.Dataset.list_files(absolute_dataset_path, shuffle=False)
    raw_lines_dataset = tf.data.TextLineDataset(all_files)
    return raw_lines_dataset.map(partial(parse_line, dictionary_size))
