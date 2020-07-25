import json
import os

from tensorflow import keras
from tensorflow.keras import layers

from walter_white import model


def test_should_create_input_output_with_one_hidden_layer_model():
    neural_net_config = {
        "layers": {
            "input": {
                "neurons": 25,
            },
            "stacks": [
                {
                    "neurons": 20,
                    "activationFunction": "relu",
                }
            ],
            "output": {
                "neurons": 25,
                "activationFunction": "sigmoid",
            },
        }
    }

    actual = model.autoencoder(neural_net_config)
    input_layer = keras.Input(shape=(25,), name='input')
    output_layer = layers.Dense(20, activation='relu', name='hidden')(input_layer)
    output_layer = layers.Dense(25, activation='sigmoid', name='output')(output_layer)
    expected = keras.Model(inputs=input_layer, outputs=output_layer)

    assert type(expected) == type(actual)


def test_should_create_complex_layer_model():
    neural_net_config = {
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

    actual = model.autoencoder(neural_net_config)
    input_layer = keras.Input(shape=(100,), name='input')
    output_layer = layers.Dense(80, activation='relu')(input_layer)
    output_layer = layers.Dense(50, activation='relu')(output_layer)
    output_layer = layers.Dense(20, activation='tanh')(output_layer)
    output_layer = layers.Dense(50, activation='relu')(output_layer)
    output_layer = layers.Dense(80, activation='relu')(output_layer)
    output_layer = layers.Dense(100, activation='sigmoid', name='output')(output_layer)
    expected = keras.Model(inputs=input_layer, outputs=output_layer)

    assert type(expected) == type(actual)


def test_should_persist_mae_to_file():
    metrics_file_path = './tests/resources/metrics.json'
    if os.path.exists(metrics_file_path):
        os.remove(metrics_file_path)

    history = {
        'loss': [0.7043520392793597, 0.702940134707829],
        'mean_absolute_error': [0.42077133, 0.42006454],
        'val_loss': [0.7049564123153687, 0.7046186923980713],
        'val_mean_absolute_error': [0.42031777, 0.42013824]
    }

    model.store_metrics(history, metrics_file_path)

    with open(metrics_file_path) as metrics_file:
        expected = history
        actual = json.load(metrics_file)

        assert expected == actual
