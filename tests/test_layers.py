# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
import keras
import numpy as np
import pytest

from radar_raman.layers import EdgePad1D, ExpandLastDim, ResidualSubtract


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8, 33, 64, 128])
def test_edgepad_length_and_values(size):
    x = np.arange(10, dtype="float32").reshape(1, 10, 1)
    y = keras.ops.convert_to_numpy(EdgePad1D(size)(x))
    assert y.shape == (1, 10 + size - 1, 1)
    left = size // 2
    np.testing.assert_array_equal(y[0, :left, 0], 0)            # first sample repeated
    np.testing.assert_array_equal(y[0, left:left + 10, 0], x[0, :, 0])
    np.testing.assert_array_equal(y[0, left + 10:, 0], 9)       # last sample repeated


@pytest.mark.parametrize("size", [3, 8, 64])
def test_edgepad_keeps_length_through_valid_conv(size):
    inp = keras.Input((None, 1))
    out = keras.layers.Conv1D(2, size, padding="valid")(EdgePad1D(size)(inp))
    m = keras.Model(inp, out)
    assert m.predict(np.zeros((2, 37, 1)), verbose=0).shape == (2, 37, 2)


def test_edgepad_config_roundtrip():
    layer = EdgePad1D(33)
    clone = EdgePad1D.from_config(layer.get_config())
    assert clone.size == 33 and clone.left == layer.left and clone.right == layer.right


def test_edgepad_invalid():
    with pytest.raises(ValueError):
        EdgePad1D(0)


def test_expand_and_subtract():
    a = np.ones((2, 5), "float32")
    assert keras.ops.convert_to_numpy(ExpandLastDim()(a)).shape == (2, 5, 1)
    r = ResidualSubtract()([a * 5, a, a * 2])
    np.testing.assert_allclose(keras.ops.convert_to_numpy(r), 2.0)
