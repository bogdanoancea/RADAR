# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Loss functions and metrics used to train the RADAR networks.

These are the functions of the original implementation, re-written with
``keras.ops`` so that they work with Keras 3 and can be serialised together
with a saved model.
"""

from __future__ import annotations

import keras
from keras import ops

__all__ = ["root_mean_squared_error", "r_square", "r_square_loss", "joined_loss"]


@keras.saving.register_keras_serializable(package="radar")
def root_mean_squared_error(y_true, y_pred):
    """Root mean squared error over all elements of the batch.

    Loss of the *standard* model.
    """
    y_true = ops.cast(y_true, y_pred.dtype)
    return ops.sqrt(ops.mean(ops.square(y_pred - y_true)))


@keras.saving.register_keras_serializable(package="radar")
def r_square(y_true, y_pred):
    """Coefficient of determination R^2 over all elements of the batch."""
    y_true = ops.cast(y_true, y_pred.dtype)
    ss_res = ops.sum(ops.square(y_true - y_pred))
    ss_tot = ops.sum(ops.square(y_true - ops.mean(y_true)))
    return 1.0 - ss_res / (ss_tot + keras.config.epsilon())


@keras.saving.register_keras_serializable(package="radar")
def r_square_loss(y_true, y_pred):
    """``1 - R^2`` = SS_res / SS_tot (minimised, lower bound 0)."""
    y_true = ops.cast(y_true, y_pred.dtype)
    ss_res = ops.sum(ops.square(y_true - y_pred))
    ss_tot = ops.sum(ops.square(y_true - ops.mean(y_true)))
    return ss_res / (ss_tot + keras.config.epsilon())


@keras.saving.register_keras_serializable(package="radar")
def joined_loss(y_true, y_pred):
    """RMSE + (1 - R^2). Loss of the *ensemble* model."""
    return root_mean_squared_error(y_true, y_pred) + r_square_loss(y_true, y_pred)
