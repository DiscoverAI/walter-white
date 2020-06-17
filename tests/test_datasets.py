import numpy as np
import numpy.testing as npt

from walter_white import datasets


def test_should_load_dataset_from_one_file():
    actual_dataset = datasets.load_dataset('tests/resources/test/part-1.csv', 1)
    actual = next(iter(actual_dataset.batch(3)))

    npt.assert_array_equal(actual, np.array([
        [25.0, 26.0, 22.0],
        [25.0, 25.0, 1.0],
        [25.0, 21.0, 2.0]
    ], dtype=float))


def test_should_load_dataset_from_directory():
    actual_dataset = datasets.load_dataset('tests/resources/test/*.csv', 10)
    actual = next(iter(actual_dataset.batch(8)))

    npt.assert_array_equal(actual, np.array([
        [2.5, 2.6, 2.2],
        [2.5, 2.5, 0.1],
        [2.5, 2.1, 0.2],
        [2.5, 2.5, 2.3],
        [0.1, 2.6, 2.2],
        [3.3, 2.5, 0.1],
        [4.2, 2.1, 0.2],
        [0.2, 2.5, 2.3],
    ]))


def test_should_return_dictionary_size_with_one_file_existing():
    actual = datasets.dictionary_size('tests/resources/dictionary-one-file')
    expected = 26.0

    assert expected == actual


def test_should_return_dictionary_size_with_n_file_existing():
    actual = datasets.dictionary_size('tests/resources/dictionary-two-files')
    expected = 9

    assert expected == actual
