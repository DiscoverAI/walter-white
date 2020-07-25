import json
import logging
import os
from functools import reduce

import boto3
import tensorflow as tf

from walter_white.numpy_encoder import NumpyArrayEncoder

LOG = logging.getLogger(__name__)


def _compile_layer(acc, layer_config):
    return tf.keras.layers.Dense(
        layer_config['neurons'],
        activation=layer_config['activationFunction']
    )(acc)


def autoencoder(neural_net_config):
    nn_layers = neural_net_config['layers']
    input_layer = tf.keras.Input(shape=(nn_layers['input']['neurons'],), name='input')
    stacks = reduce(_compile_layer, nn_layers['stacks'], input_layer)
    output_layer = tf.keras.layers.Dense(
        nn_layers['output']['neurons'],
        activation=nn_layers['output']['activationFunction'],
        name='output'
    )(stacks)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


def compile_model(neural_network_config):
    nn_architecture = autoencoder(neural_network_config)
    nn_architecture.summary(print_fn=LOG.info)
    nn_architecture.compile(
        loss=neural_network_config['lossFunction'],
        metrics=['mean_absolute_error'],
        optimizer=neural_network_config['optimizer'],
    )
    return nn_architecture


def _upload_folder(bucket, remote_output_folder, local_folder_path):
    for path, _subdirs, files in os.walk(local_folder_path):
        directory_name = path.replace(local_folder_path, "")
        for file in files:
            remote_key = remote_output_folder + directory_name + '/' + file
            local_path = os.path.join(path, file)
            bucket.upload_file(local_path, remote_key.replace('//', '/'))


def store_metrics(history, local_path):
    with open(local_path, 'w') as metrics_file:
        json.dump(history, metrics_file, cls=NumpyArrayEncoder)


def persist_model(training_history, model, datalake, output_folder):
    metrics_file_path = './metrics.json'
    store_metrics(training_history, metrics_file_path)
    local_path = './generative_model/'
    model.save(local_path)
    s_3 = boto3.resource('s3')
    bucket = s_3.Bucket(datalake)
    _upload_folder(bucket, output_folder + 'tensorboard/', './resources/tensorboard')
    _upload_folder(bucket, output_folder + 'model/', local_path)
    bucket.upload_file(metrics_file_path, output_folder + 'metrics.json')
    return output_folder + 'model/'
