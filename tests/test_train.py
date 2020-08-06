from walter_white import train


def test_should_format_2_layers_neural_network_config():
    config = {
        "optimizer": "adadelta",
        "layers": {
            "input": {
                "neurons": 57,
            },
            "stacks": [],
            "output": {
                "neurons": 57,
                "activationFunction": "sigmoid",
            },
        }
    }

    actual = train.format_layers(config["layers"])
    expected = [
        {'key': 'input', 'value': {"neurons": 57}},
        {
            'key': 'output',
            'value': {
                "neurons": 57,
                "activationFunction": "sigmoid",
            }
        }
    ]

    assert expected == actual


def test_should_format_3_layers_neural_network_config():
    config = {
        "optimizer": "adadelta",
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

    actual = train.format_layers(config["layers"])
    expected = [
        {'key': 'input', 'value': {"neurons": 57}},
        {'key': 'h1', 'value': {
            "neurons": 30,
            "activationFunction": "sigmoid"}
         },
        {
            'key': 'output',
            'value': {
                "neurons": 57,
                "activationFunction": "sigmoid",
            }
        }
    ]

    assert expected == actual


def test_should_format_n_layers_neural_network_config():
    config = {
        "optimizer": "adadelta",
        "layers": {
            "input": {
                "neurons": 13,
            },
            "stacks": [
                {
                    "neurons": 30,
                    "activationFunction": "sigmoid",
                }, {
                    "neurons": 14,
                    "activationFunction": "tanh",
                },
            ],
            "output": {
                "neurons": 41,
                "activationFunction": "relu",
            },
        }
    }

    actual = train.format_layers(config["layers"])
    expected = [
        {'key': 'input', 'value': {"neurons": 13}},
        {'key': 'h1', 'value': {
            "neurons": 30,
            "activationFunction": "sigmoid"}
         },
        {'key': 'h2', 'value': {
            "neurons": 14,
            "activationFunction": "tanh",
        }
         },
        {
            'key': 'output',
            'value': {
                "neurons": 41,
                "activationFunction": "relu",
            }
        }
    ]

    assert expected == actual

