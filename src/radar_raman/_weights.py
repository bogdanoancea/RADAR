# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Loading of weight files (pre-trained, legacy Keras 2 and Keras 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import h5py
import keras

from .models import LEGACY_LAYER_NAMES

__all__ = ["pretrained_weights_path", "load_legacy_weights", "is_legacy_weight_file"]

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

#: Pre-trained weight files distributed with the package.
PRETRAINED_FILES: Dict[str, str] = {
    "standard": "standard.h5",
    "ensemble": "ensemble.h5",
}

#: How the bundled weights were trained (original training scripts, Keras 2.7).
PRETRAINED_TRAINING = {
    "epochs": 12,
    "steps_per_epoch": 200_000,
    "batch_size": 100,
    "spectra_seen": 240_000_000,
    "length": 750,
    "learning_rate": "original 12-level schedule (1.05e-3 -> 1.5e-6)",
    "data": "synthetic stream of the original generator (legacy_scaling=True behaviour)",
    "trained_with": "Keras 2.7 / TensorFlow 2 (original scripts by J. Sjöberg)",
}


def pretrained_weights_path(architecture: str) -> Path:
    """Path of the bundled pre-trained weights for ``architecture``."""
    try:
        return _WEIGHTS_DIR / PRETRAINED_FILES[architecture]
    except KeyError:
        raise ValueError(
            f"No pre-trained weights for {architecture!r}; available: {sorted(PRETRAINED_FILES)}"
        ) from None


def is_legacy_weight_file(path) -> bool:
    """True if ``path`` is a Keras 2 (``model.save_weights('x.h5')``) weight file."""
    try:
        with h5py.File(path, "r") as f:
            return "layer_names" in f.attrs
    except OSError:
        return False


def _decode(x):
    return x.decode("utf8") if isinstance(x, bytes) else x


def load_legacy_weights(model: keras.Model, path, architecture: str) -> keras.Model:
    """Load a Keras 2 ``.h5`` weight file written by the original RADAR scripts.

    The original files identify layers by automatically generated names
    (``conv1d_4``, ``batch_normalization_7``, ...). This function maps them to
    the named layers of :func:`radar_raman.models.build_model` using
    :data:`radar_raman.models.LEGACY_LAYER_NAMES`, and checks that every weight has
    the expected shape.

    Parameters
    ----------
    model : keras.Model
        A model built with ``build_model(architecture)`` (default
        hyper-parameters).
    path : str or Path
        Weight file, e.g. ``Models/ensembleModelRes.h5`` from the original
        repository.
    architecture : {"standard", "ensemble"}
        Architecture the weights belong to.
    """
    mapping = LEGACY_LAYER_NAMES[architecture]
    with h5py.File(path, "r") as f:
        stored = {_decode(n) for n in f.attrs["layer_names"]}
        for new_name, old_name in mapping.items():
            if old_name not in stored:
                raise ValueError(
                    f"Layer {old_name!r} not found in {path}; is this a "
                    f"{architecture!r} weight file?"
                )
            group = f[old_name]
            names = [_decode(n) for n in group.attrs["weight_names"]]
            values = [group[n][()] for n in names]
            layer = model.get_layer(new_name)
            expected = [tuple(w.shape) for w in layer.weights]
            got = [tuple(v.shape) for v in values]
            if expected != got:
                raise ValueError(
                    f"Shape mismatch for layer {new_name!r} (file layer {old_name!r}): "
                    f"model expects {expected}, file has {got}. The model was probably "
                    "built with non-default hyper-parameters."
                )
            layer.set_weights(values)
    return model
