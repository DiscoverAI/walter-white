from functools import reduce

import tensorflow as tf


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
