#!/usr/bin/env python3
import logging
import os

import tensorflow as tf

from walter_white import model, datasets

LOG = logging.getLogger(__name__)
NN_CONF = {
    "optimizer": "adadelta",
    "lossFunction": "binary_crossentropy",
    "epochs": 1,
    "batchSize": 10000,
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
        tf.keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None),
        tf.keras.callbacks.TensorBoard(log_dir='resources/tensorboard', update_freq='batch'),
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
    # datasets.download_s3_folder(datalake, 'pinkman/dictionary.csv', 'dictionary')
    # datasets.download_s3_folder(datalake, 'pinkman/test.csv', './test')
    # datasets.download_s3_folder(datalake, 'pinkman/train.csv', './train')
    LOG.info('Done downloading datasets')

    dictionary_size = datasets.calculate_dictionary_size('./dictionary')
    LOG.info('Dictionary size of: %s', dictionary_size)
    train_dataset = datasets.load_dataset('./train/*.csv', dictionary_size, datasets.MAX_SMILE_SIZE)
    test_dataset = datasets.load_dataset('./test/*.csv', dictionary_size, datasets.MAX_SMILE_SIZE)
    LOG.info('Done loading datasets')

    LOG.info('Start building model')
    nn_model = model.compile_model(NN_CONF)
    LOG.info('Done building model')

    LOG.info('Start training model')
    history = train(NN_CONF, nn_model, train_dataset, test_dataset)
    LOG.info('Done training model')

    LOG.info('Start persisting model')
    model.persist_model(history.history, nn_model, datalake, 'walter_white/')
    LOG.info('Done persisting model')
