import csv
import glob
import logging
import os
from functools import partial, reduce

import boto3
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


def read_lines(acc, file_path):
    with open(file_path, 'r') as dictionary_file:
        dictionary_reader = csv.reader(dictionary_file, delimiter=',')
        return acc + list(dictionary_reader)


def dictionary_size(dictionary_path):
    absolute_dataset_path = os.path.realpath(dictionary_path + '/*.csv')
    dictionary_files = glob.glob(absolute_dataset_path)
    lines = reduce(read_lines, dictionary_files, [])
    return len(lines)


def count_dataset_size(dataset):
    return dataset.reduce(tf.constant(0), lambda x, _: x + 1).numpy()


def download_s3_folder(datalake, folder_name, local_path):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(datalake)
    for s3_object in bucket.objects.filter(Prefix=folder_name):
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        local_file_name = s3_object.key.replace(folder_name, local_path)
        bucket.download_file(s3_object.key, local_file_name)
