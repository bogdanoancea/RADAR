# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Training utilities: learning-rate schedule and fitting loop."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import keras
import numpy as np

from .generator import SyntheticDataset, generate_dataset, make_tf_dataset

__all__ = ["ORIGINAL_LR_VALUES", "original_learning_rate_schedule", "fit"]

#: The 12 learning-rate levels of the original piecewise-constant schedule.
ORIGINAL_LR_VALUES = [
    v * 1.5
    for v in (7e-4, 5e-4, 3e-4, 1e-4, 7e-5, 5e-5, 3e-5, 1e-5, 7e-6, 5e-6, 3e-6, 1e-6)
]


def original_learning_rate_schedule(total_steps: int):
    """Original piecewise-constant decay, stretched to ``total_steps``.

    The pre-trained models were trained for 12 epochs x 200 000 steps of 100
    spectra (240 million spectra), lowering the learning
    rate every 200 000 steps through :data:`ORIGINAL_LR_VALUES` (1.05e-3 down to
    1.5e-6). This function keeps the 12 levels and places the 11 boundaries
    at ``k * total_steps / 12``; with ``total_steps = 2_400_000`` it is
    identical to the original schedule.
    """
    total_steps = int(total_steps)
    if total_steps < 12:
        return ORIGINAL_LR_VALUES[0]
    boundaries = [int(round(total_steps * k / 12)) for k in range(1, 12)]
    return keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, ORIGINAL_LR_VALUES)


def _as_keras_data(data):
    if data is None:
        return None
    if isinstance(data, SyntheticDataset):
        return data.as_keras()
    X, y = data
    X = np.asarray(X, dtype="float32")
    if X.ndim == 2:
        X = X[..., None]
    if isinstance(y, dict):
        from .models import COMPONENTS

        y = [y[c] for c in COMPONENTS]
    return X, [np.asarray(t, dtype="float32") for t in y]


def fit(
    model: keras.Model,
    data: Union[None, SyntheticDataset, tuple] = None,
    *,
    epochs: int = 10,
    steps_per_epoch: int = 1000,
    batch_size: int = 100,
    length: int = 750,
    min_width: int = 5,
    max_width: int = 256,
    max_num_peaks: int = 5,
    increase_noise: bool = False,
    legacy_scaling: bool = False,
    validation_data: Union[None, str, SyntheticDataset, tuple] = "synthetic",
    validation_size: int = 1000,
    validation_length: int = 2000,
    seed: Optional[int] = 0,
    callbacks: Optional[Sequence[keras.callbacks.Callback]] = None,
    checkpoint_path: Optional[str] = None,
    verbose: int = 1,
) -> keras.callbacks.History:
    """Fit a compiled RADAR model.

    See :meth:`radar.Decomposer.train` for the meaning of the arguments.
    """
    if validation_data == "synthetic":
        validation_data = generate_dataset(
            validation_size,
            length=validation_length,
            min_width=5,
            max_width=300,
            max_num_peaks=5,
            increase_noise=increase_noise,
            legacy_scaling=legacy_scaling,
            seed=None if seed is None else seed + 12345,
        )
    validation_data = _as_keras_data(validation_data)

    cbs: List[keras.callbacks.Callback] = list(callbacks or [])
    if checkpoint_path is not None:
        cbs.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor="val_loss" if validation_data is not None else "loss",
                save_best_only=True,
            )
        )

    if data is None:
        ds = make_tf_dataset(
            length=length,
            batch_size=batch_size,
            min_width=min_width,
            max_width=max_width,
            max_num_peaks=max_num_peaks,
            increase_noise=increase_noise,
            legacy_scaling=legacy_scaling,
            seed=seed,
        )
        return model.fit(
            ds,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=cbs,
            shuffle=False,  # freshly generated stream
            verbose=verbose,
        )

    X, y = _as_keras_data(data)
    return model.fit(
        X,
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=cbs,
        shuffle=True,
        verbose=verbose,
    )
