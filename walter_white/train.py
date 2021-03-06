#!/usr/bin/env python3
import logging
import os
import sys
from functools import reduce

import mlflow
import tensorflow as tf

from walter_white import model, datasets

LOG = logging.getLogger(__name__)
EXPERIMENT_NAME = 'sars-cov-2'
JOB = 'walter_white'
NN_CONF = {
    "optimizer": "adadelta",
    "lossFunction": "mean_absolute_error",
    "epochs": 1,
    "batchSize": 100000,
    "layers": {
        "input": {
            "neurons": 57,
        },
        "stacks": [
            {
                "neurons": 30,
                "activationFunction": "sigmoid",
            },
        ],
        "output": {
            "neurons": 57,
            "activationFunction": "sigmoid",
        },
    }
}


def train(neural_network_config, neural_network_model, train_ds, test_ds):
    batch_size = neural_network_config['batchSize']
    epochs = neural_network_config['epochs']

    train_dataset_size = 1584663
    train_dataset_batches = int(train_dataset_size / batch_size)
    batched_train_dataset = train_ds.batch(batch_size).repeat()

    test_dataset_size = 176074
    test_dataset_batches = int(test_dataset_size / batch_size)
    batched_test_dataset = test_ds.batch(batch_size).repeat()

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='resources/tensorboard', update_freq='batch'),
    ]
    gpu_count = str(len(tf.config.experimental.list_physical_devices('GPU')))
    LOG.info('Num GPUs Available: %s', gpu_count)

    return neural_network_model.fit(
        batched_train_dataset,
        epochs=epochs,
        steps_per_epoch=train_dataset_batches,
        validation_data=batched_test_dataset,
        validation_steps=test_dataset_batches,
        shuffle=False,
        callbacks=callbacks,
    )


def log_nn_config(nn_config):
    mlflow.log_param('optimizer', nn_config['optimizer'])
    mlflow.log_param('lossFunction', nn_config['lossFunction'])
    mlflow.log_param('epochs', nn_config['epochs'])
    mlflow.log_param('batchSize', nn_config['batchSize'])


def log_metrics(training_history, normalization_factor):
    mean_absolute_error = training_history['val_mean_absolute_error'][-1]
    mlflow.log_metric('mae', mean_absolute_error * normalization_factor)
    mlflow.log_metric('mae_normalised', mean_absolute_error * normalization_factor)


def log_layers(layers):
    for layer in layers:
        mlflow.log_param(layer["key"] + "Layer", layer["value"])


def format_layer(key, layer):
    return {"key": key, "value": layer}


def format_hidden_layer(acc, hidden_layer):
    return {
        'index': acc['index'] + 1,
        'stacks': acc['stacks'] + [format_layer("h" + str(acc['index']), hidden_layer)]
    }


def format_layers(config):
    formatted_input_layer = [format_layer("input", config['input'])]
    formatted_hidden_layers = reduce(format_hidden_layer, config['stacks'],
                                     {'index': 1, 'stacks': []})
    formatted_output_layer = [format_layer("output", config['output'])]
    return formatted_input_layer + formatted_hidden_layers['stacks'] + formatted_output_layer


if __name__ == '__main__':
    LOG.info('Start connecting to mlFlow instance')
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run(run_name=JOB)
    LOG.info('Done connecting to mlFlow instance')

    try:
        LOG.info('Start loading datasets')

        LOG.info('Start downloading datasets')
        datalake = os.environ['DATALAKE'].replace('s3://', '')
        mlflow.log_param('input', os.environ['DATALAKE'] + '/pinkman/')
        datasets.download_s3_folder(datalake, 'pinkman/dictionary.csv', 'dictionary')
        datasets.download_s3_folder(datalake, 'pinkman/test.csv', './test')
        datasets.download_s3_folder(datalake, 'pinkman/train.csv', './train')
        LOG.info('Done downloading datasets')

        dictionary_size = datasets.calculate_dictionary_size('./dictionary')
        LOG.info('Dictionary size of: %s', dictionary_size)
        mlflow.log_param('dictionarySize', dictionary_size)
        train_dataset = datasets.load_dataset(
            './train/*.csv',
            dictionary_size,
            datasets.MAX_SMILE_SIZE
        )
        test_dataset = datasets.load_dataset(
            './test/*.csv',
            dictionary_size,
            datasets.MAX_SMILE_SIZE
        )
        LOG.info('Done loading datasets')

        LOG.info('Start building model')
        nn_model = model.compile_model(NN_CONF)
        log_nn_config(NN_CONF)
        log_layers(format_layers(NN_CONF["layers"]))
        LOG.info('Done building model')

        LOG.info('Start training model')
        history = train(NN_CONF, nn_model, train_dataset, test_dataset)
        log_metrics(history.history, dictionary_size)
        LOG.info('Done training model')

        LOG.info('Start persisting model')
        MODEL_PATH = model.persist_model(history.history, nn_model, datalake, 'walter_white/')
        mlflow.log_param('output', os.environ['DATALAKE'] + '/' + MODEL_PATH)
        LOG.info('Done persisting model')
        mlflow.end_run('FINISHED')
    except Exception as exception:
        LOG.error('could not finish run successfully')
        mlflow.end_run('FAILED')
        sys.exit(exception)
