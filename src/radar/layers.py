# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Custom Keras layers used by the RADAR networks.

All layers are registered as Keras serializable objects (package ``"radar"``),
so models that use them can be saved to and loaded from the native ``.keras``
format without passing ``custom_objects``.
"""

from __future__ import annotations

import keras
from keras import ops

__all__ = ["EdgePad1D", "CustomPad", "ExpandLastDim", "ResidualSubtract"]


@keras.saving.register_keras_serializable(package="radar")
class EdgePad1D(keras.layers.Layer):
    """Pad a sequence by repeating its first and last samples.

    The RADAR networks use ``"valid"`` convolutions on edge-padded inputs so
    that every convolution keeps the spectrum length unchanged, independent of
    the kernel size, and so that the spectrum edges are not pulled towards zero
    (which zero-padding would do).

    For a kernel/pool size ``k`` the layer adds ``k // 2`` copies of the first
    sample on the left and ``k // 2 - 1 + k % 2`` copies of the last sample on
    the right, i.e. ``k - 1`` samples in total. A following ``"valid"`` layer
    with window ``k`` then returns a sequence of the original length.

    Parameters
    ----------
    size : int
        Size of the window of the following convolution / pooling layer.

    Input shape
    -----------
    ``(batch, length, channels)``

    Output shape
    ------------
    ``(batch, length + size - 1, channels)``

    Notes
    -----
    This is the ``CustomPad`` layer of the original implementation by
    J. Sjöberg. The original ``get_config`` stored ``size // 2`` instead of
    ``size``, which made re-created layers pad incorrectly; this version stores
    the true ``size``.
    """

    def __init__(self, size: int, **kwargs):
        super().__init__(**kwargs)
        size = int(size)
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        self.size = size
        self.left = size // 2
        self.right = size // 2 - 1 + size % 2

    def call(self, inputs):
        parts = []
        if self.left > 0:
            parts.append(ops.repeat(inputs[:, :1, :], self.left, axis=1))
        parts.append(inputs)
        if self.right > 0:
            parts.append(ops.repeat(inputs[:, -1:, :], self.right, axis=1))
        return ops.concatenate(parts, axis=1) if len(parts) > 1 else inputs

    def compute_output_shape(self, input_shape):
        length = input_shape[1]
        new_len = None if length is None else length + self.left + self.right
        return (input_shape[0], new_len, input_shape[2])

    def get_config(self):
        config = super().get_config()
        config.update({"size": self.size})
        return config


#: Backwards-compatible alias with the name used in the original scripts.
CustomPad = EdgePad1D


@keras.saving.register_keras_serializable(package="radar")
class ExpandLastDim(keras.layers.Layer):
    """Append a trailing axis of size one (``x[..., None]``)."""

    def call(self, inputs):
        return ops.expand_dims(inputs, -1)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape) + (1,)


@keras.saving.register_keras_serializable(package="radar")
class ResidualSubtract(keras.layers.Layer):
    """Return ``inputs[0] - inputs[1] - ... - inputs[n]``.

    Used to remove already-estimated components from the input spectrum
    (e.g. *noise = spectrum - baseline - cosmic rays - peaks* in the standard
    model). Replaces the ``Add()([x, -a, -b])`` construction of the original
    code, which is not serialisable in Keras 3.
    """

    def call(self, inputs):
        out = inputs[0]
        for x in inputs[1:]:
            out = out - x
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0]
