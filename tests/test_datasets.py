import numpy as np
import numpy.testing as npt

from walter_white import datasets


def test_should_load_dataset():
    actual_dataset = datasets.load_dataset('tests/resources/test.csv')
    actual = next(iter(actual_dataset.batch(3)))

    npt.assert_array_equal(actual, np.array([
        [25.0, 26.0, 22.0],
        [25.0, 25.0, 1.0],
        [25.0, 21.0, 2.0]
    ], dtype=float))
