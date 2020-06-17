#!/usr/bin/env python3
import logging
import os

import tensorflow as tf

from walter_white import model, datasets

LOG = logging.getLogger(__name__)
NN_CONF = {
    "optimizer": "adam",
    "lossFunction": "binary_crossentropy",
    "epochs": 36,
    "batchSize": 64,
    "layers": {
        "input": {
            "neurons": 100,
        },
        "stacks": [
            {
                "neurons": 80,
                "activationFunction": "relu",
            },
            {
                "neurons": 50,
                "activationFunction": "relu",
            },
            {
                "neurons": 20,
                "activationFunction": "tanh",
            },
            {
                "neurons": 50,
                "activationFunction": "relu",
            },
            {
                "neurons": 80,
                "activationFunction": "relu",
            },
        ],
        "output": {
            "neurons": 100,
            "activationFunction": "sigmoid",
        },
    }
}


def train(neural_network_config, neural_network_model, train_dataset, test_dataset):
    batch_size = neural_network_config['batchSize']
    epochs = neural_network_config['epochs']

    train_dataset_size = datasets.calculate_dataset_size(train_dataset)
    train_dataset_batches = int(train_dataset_size / batch_size)
    batched_train_dataset = train_dataset.batch(batch_size).repeat()

    test_dataset_size = datasets.calculate_dataset_size(test_dataset)
    test_dataset_batches = int(test_dataset_size / batch_size)
    batched_test_dataset = test_dataset.batch(batch_size).repeat()

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='resources/tensorboard')
    ]

    return neural_network_model.fit(
        batched_train_dataset,
        epochs=epochs,
        steps_per_epoch=train_dataset_batches,
        validation_data=batched_test_dataset,
        validation_steps=test_dataset_batches,
        shuffle=False,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    LOG.info('Start loading datasets')

    LOG.info('Start downloading datasets')
    datalake = os.environ['DATALAKE'].replace('s3://', '')
    datasets.download_s3_folder(datalake, 'pinkman/dictionary.csv', 'dictionary')
    datasets.download_s3_folder(datalake, 'pinkman/test.csv', './test')
    datasets.download_s3_folder(datalake, 'pinkman/train.csv', './train')
    LOG.info('Done downloading datasets')

    dictionary_size = datasets.calculate_dictionary_size('./dictionary')
    LOG.info('Dictionary size of: %s', dictionary_size)
    train_dataset = datasets.load_dataset('./train', dictionary_size, datasets.MAX_SMILE_SIZE)
    test_dataset = datasets.load_dataset('./test', dictionary_size, datasets.MAX_SMILE_SIZE)
    LOG.info('Done loading datasets')

    LOG.info('Start building model')
    nn_model = model.compile_model(NN_CONF)
    LOG.info('Done building model')

    LOG.info('Start training model')
    train(NN_CONF, nn_model, train_dataset, test_dataset)
    LOG.info('Done training model')
