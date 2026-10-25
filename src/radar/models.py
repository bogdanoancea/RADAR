# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""The two RADAR convolutional architectures.

Both networks take a batch of (min-max normalised) spectra of shape
``(batch, length, 1)`` -- any ``length`` is accepted -- and return a list of
four tensors of shape ``(batch, length)``, in this order:

``[baseline, cosmic_rays, noise, peaks]``

*Standard model* (``"standard"``)
    A single encoder with a learned "key" vector (global max-pooled
    convolution) that is combined with the input through an outer product,
    processed by 2-D convolutions and decoded by 1-D convolutions. Three heads
    estimate baseline, cosmic rays and peaks; the noise is the remainder
    ``input - baseline - cosmic_rays - peaks``.

*Ensemble model* (``"ensemble"``)
    A two-stage network. Stage 1 (same structure as the standard model, with
    fewer filters) estimates baseline and cosmic rays, which are subtracted
    from the input. Stage 2 processes this reduced spectrum, with a residual
    connection from stage 1, and has two heads for peaks and noise.

The architectures, hyper-parameters, and the pre-trained weights shipped with
the package are those of the original implementation by J. Sjöberg.
"""

from __future__ import annotations

from typing import Callable, Dict

import keras
import numpy as np
from keras import layers
from keras.regularizers import L1L2, l2

from .layers import EdgePad1D, ExpandLastDim, ResidualSubtract
from .losses import joined_loss, r_square, root_mean_squared_error

__all__ = [
    "COMPONENTS",
    "ARCHITECTURES",
    "build_standard_model",
    "build_ensemble_model",
    "build_model",
    "compile_model",
    "LEGACY_LAYER_NAMES",
]

#: Order of the model outputs.
COMPONENTS = ("baseline", "cosmic_rays", "noise", "peaks")


def _conv1d(x, filters, kernel_size, name, l1_param, l2_param, head=False, pad=True):
    """Edge-padded 'valid' Conv1D with the original regularisation."""
    if pad:
        x = EdgePad1D(kernel_size, name=f"{name}_pad")(x)
    return layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding="valid",
        kernel_regularizer=L1L2(l1=1e-5 if head else l1_param, l2=l2_param),
        bias_regularizer=l2(l2_param),
        activity_regularizer=l2(l2_param),
        name=name,
    )(x)


def _conv2d(x, filters, kernel_size, strides, name, l1_param, l2_param):
    return layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_regularizer=L1L2(l1=l1_param, l2=l2_param),
        bias_regularizer=l2(l2_param),
        activity_regularizer=l2(l2_param),
        name=name,
    )(x)


def _bn_act(x, name):
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    # negative_slope=0.3 is the Keras 2 default used when the weights were trained
    return layers.LeakyReLU(negative_slope=0.3, name=f"{name}_act")(x)


def _attention_block(signal, features, prefix, scaler, dim_red_size, l1_param, l2_param):
    """Key-vector block: outer product of the signal with a learned key.

    Returns the output of the second 2-D convolution *before* its batch
    normalisation (needed as residual by the ensemble model).
    """
    key = _conv1d(features, dim_red_size, 64, f"{prefix}key_conv", l1_param, l2_param)
    key = layers.GlobalMaxPooling1D(name=f"{prefix}key_pool")(key)
    key = ExpandLastDim(name=f"{prefix}key_expand")(key)  # (B, D, 1)
    t = layers.Dot(axes=(2, 2), name=f"{prefix}outer")([signal, key])  # (B, L, D)
    t = ExpandLastDim(name=f"{prefix}outer_expand")(t)  # (B, L, D, 1)
    k2 = int(dim_red_size / 10)
    stride = (1, int(np.sqrt(dim_red_size)))
    t = _conv2d(t, scaler, (32, k2), stride, f"{prefix}conv2d_1", l1_param, l2_param)
    t = _bn_act(t, f"{prefix}conv2d_1")
    t = _conv2d(t, scaler, (16, k2), stride, f"{prefix}conv2d_2", l1_param, l2_param)
    return t


def _check_dim_red(dim_red_size):
    root = int(np.sqrt(dim_red_size))
    if root * root != dim_red_size:
        raise ValueError(f"dim_red_size must be a perfect square, got {dim_red_size}")


def build_standard_model(
    scaler: int = 20,
    dim_red_size: int = 16**2,
    l1_param: float = 1e-6,
    l2_param: float = 1e-6,
    seed: int | None = 0,
    name: str = "radar_standard",
) -> keras.Model:
    """Build the (uncompiled) *standard* RADAR network.

    Parameters
    ----------
    scaler : int
        Number of filters of the hidden convolutions (20 in the pre-trained model).
    dim_red_size : int
        Length of the learned key vector; must be a perfect square
        (256 in the pre-trained model).
    l1_param, l2_param : float
        L1/L2 regularisation strengths of the hidden layers.
    seed : int or None
        Seed for the weight initialisation (``None`` = do not seed).
    name : str
        Model name.

    Returns
    -------
    keras.Model
        Model mapping ``(batch, length, 1)`` to
        ``[baseline, cosmic_rays, noise, peaks]``, each ``(batch, length)``.
    """
    _check_dim_red(dim_red_size)
    if seed is not None:
        keras.utils.set_random_seed(seed)

    inp = keras.Input(shape=(None, 1), name="spectrum")
    t = _conv1d(inp, scaler, 128, "enc_conv1", l1_param, l2_param)
    t = _bn_act(t, "enc_conv1")
    t = _attention_block(inp, t, "", scaler, dim_red_size, l1_param, l2_param)
    t = _bn_act(t, "conv2d_2")
    t = layers.Reshape((-1, scaler), name="to_1d")(t)
    t = _conv1d(t, scaler, 128, "dec_conv1", l1_param, l2_param)
    t = _bn_act(t, "dec_conv1")
    t = _conv1d(t, scaler, 64, "dec_conv2", l1_param, l2_param)
    t = _bn_act(t, "dec_conv2")

    # Baseline head: wide kernel followed by smoothing
    bl = _conv1d(t, 1, 64, "baseline_conv", l1_param, l2_param, head=True)
    bl = layers.Activation("relu", name="baseline_relu1")(bl)
    bl = EdgePad1D(33, name="baseline_pool_pad")(bl)
    bl = layers.AveragePooling1D(33, 1, padding="valid", name="baseline_pool")(bl)
    bl = layers.Activation("relu", name="baseline_relu2")(bl)

    # Cosmic-ray head: very narrow kernel
    cr = _conv1d(t, 1, 3, "cosmic_ray_conv", l1_param, l2_param, head=True)
    cr = layers.Activation("relu", name="cosmic_ray_relu")(cr)

    # Peak head: small kernel to keep sharp peaks
    pk = _conv1d(t, 1, 8, "peak_conv", l1_param, l2_param, head=True)
    pk = layers.Activation("relu", name="peak_relu1")(pk)
    pk = EdgePad1D(4, name="peak_pool_pad")(pk)
    pk = layers.AveragePooling1D(4, 1, padding="valid", name="peak_pool")(pk)
    pk = layers.Activation("relu", name="peak_relu2")(pk)

    # Noise = what is left after removing the other three components
    nz = ResidualSubtract(name="noise_residual")([inp, bl, cr, pk])

    outputs = [
        layers.Flatten(name="baseline")(bl),
        layers.Flatten(name="cosmic_rays")(cr),
        layers.Flatten(name="noise")(nz),
        layers.Flatten(name="peaks")(pk),
    ]
    return keras.Model(inp, outputs, name=name)


def build_ensemble_model(
    scaler: int = 10,
    dim_red_size: int = 12**2,
    l1_param: float = 1e-6,
    l2_param: float = 1e-6,
    seed: int | None = 0,
    name: str = "radar_ensemble",
) -> keras.Model:
    """Build the (uncompiled) two-stage *ensemble* RADAR network.

    Parameters
    ----------
    scaler : int
        Number of filters of the hidden convolutions (10 in the pre-trained model).
    dim_red_size : int
        Length of the learned key vectors; must be a perfect square
        (144 in the pre-trained model).
    l1_param, l2_param : float
        L1/L2 regularisation strengths of the hidden layers.
    seed : int or None
        Seed for the weight initialisation (``None`` = do not seed).
    name : str
        Model name.

    Returns
    -------
    keras.Model
        Model mapping ``(batch, length, 1)`` to
        ``[baseline, cosmic_rays, noise, peaks]``, each ``(batch, length)``.
    """
    _check_dim_red(dim_red_size)
    if seed is not None:
        keras.utils.set_random_seed(seed)

    inp = keras.Input(shape=(None, 1), name="spectrum")

    # ---- Stage 1: baseline and cosmic rays --------------------------------
    t = _conv1d(inp, scaler, 128, "enc_conv1", l1_param, l2_param)
    t = _bn_act(t, "enc_conv1")
    t_res = _attention_block(inp, t, "", scaler, dim_red_size, l1_param, l2_param)
    t = _bn_act(t_res, "conv2d_2")
    t = layers.Reshape((-1, scaler), name="to_1d")(t)
    t = _conv1d(t, scaler, 128, "dec_conv1", l1_param, l2_param)
    t = _bn_act(t, "dec_conv1")
    t = _conv1d(t, scaler, 64, "dec_conv2", l1_param, l2_param)
    t = _bn_act(t, "dec_conv2")

    bl = _conv1d(t, 1, 64, "baseline_conv", l1_param, l2_param, head=True)
    bl = EdgePad1D(33, name="baseline_pool_pad")(bl)
    bl = layers.AveragePooling1D(33, 1, padding="valid", name="baseline_pool")(bl)
    bl = layers.Activation("relu", name="baseline_relu")(bl)

    cr = _conv1d(t, 1, 3, "cosmic_ray_conv", l1_param, l2_param, head=True)
    cr = layers.Activation("relu", name="cosmic_ray_relu")(cr)

    # ---- Stage 2: peaks and noise from the reduced spectrum ---------------
    reduced = ResidualSubtract(name="reduced_spectrum")([inp, bl, cr])
    t = _conv1d(reduced, scaler, 128, "s2_enc_conv1", l1_param, l2_param)
    t = _bn_act(t, "s2_enc_conv1")
    t = _attention_block(reduced, t, "s2_", scaler, dim_red_size, l1_param, l2_param)
    t = layers.Add(name="s2_residual")([t, t_res])
    t = _bn_act(t, "s2_conv2d_2")
    t = layers.Reshape((-1, scaler), name="s2_to_1d")(t)
    t = _conv1d(t, scaler, 128, "s2_dec_conv1", l1_param, l2_param)
    t = _bn_act(t, "s2_dec_conv1")
    t = _conv1d(t, scaler, 64, "s2_dec_conv2", l1_param, l2_param)
    t = _bn_act(t, "s2_dec_conv2")

    pk = _conv1d(t, 1, 8, "peak_conv", l1_param, l2_param, head=True)
    pk = EdgePad1D(4, name="peak_pool_pad")(pk)
    pk = layers.AveragePooling1D(4, 1, padding="valid", name="peak_pool")(pk)
    pk = layers.Activation("relu", name="peak_relu")(pk)

    nz = _conv1d(t, 1, 1, "noise_conv", l1_param, l2_param, head=True, pad=False)
    nz = layers.Activation("tanh", name="noise_tanh")(nz)

    outputs = [
        layers.Flatten(name="baseline")(bl),
        layers.Flatten(name="cosmic_rays")(cr),
        layers.Flatten(name="noise")(nz),
        layers.Flatten(name="peaks")(pk),
    ]
    return keras.Model(inp, outputs, name=name)


#: Registry of available architectures.
ARCHITECTURES: Dict[str, Callable[..., keras.Model]] = {
    "standard": build_standard_model,
    "ensemble": build_ensemble_model,
}


def build_model(architecture: str = "ensemble", **kwargs) -> keras.Model:
    """Build an uncompiled RADAR network by name (``"standard"`` or ``"ensemble"``)."""
    try:
        builder = ARCHITECTURES[architecture]
    except KeyError:
        raise ValueError(
            f"Unknown architecture {architecture!r}; choose from {sorted(ARCHITECTURES)}"
        ) from None
    return builder(**kwargs)


def compile_model(model: keras.Model, architecture: str, learning_rate=1e-5) -> keras.Model:
    """Compile ``model`` with the original optimiser, loss and metrics.

    * standard: Adam + RMSE on every output
    * ensemble: Adam + (RMSE + 1 - R^2) on every output, RMSE and R^2 as metrics

    ``learning_rate`` may be a float or a Keras learning-rate schedule.
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    if architecture == "standard":
        model.compile(optimizer=optimizer, loss=[root_mean_squared_error] * 4)
    elif architecture == "ensemble":
        model.compile(
            optimizer=optimizer,
            loss=[joined_loss] * 4,
            metrics=[[root_mean_squared_error, r_square] for _ in range(4)],
        )
    else:
        raise ValueError(f"Unknown architecture {architecture!r}")
    return model


# ---------------------------------------------------------------------------
# Mapping from the layer names of this implementation to the (automatic)
# layer names in the Keras 2.7 weight files of the original implementation.
# ---------------------------------------------------------------------------
_STAGE1 = {
    "enc_conv1": "conv1d",
    "enc_conv1_bn": "batch_normalization",
    "key_conv": "conv1d_1",
    "conv2d_1": "conv2d",
    "conv2d_1_bn": "batch_normalization_1",
    "conv2d_2": "conv2d_1",
    "conv2d_2_bn": "batch_normalization_2",
    "dec_conv1": "conv1d_2",
    "dec_conv1_bn": "batch_normalization_3",
    "dec_conv2": "conv1d_3",
    "dec_conv2_bn": "batch_normalization_4",
    "baseline_conv": "conv1d_4",
    "cosmic_ray_conv": "conv1d_5",
}

LEGACY_LAYER_NAMES: Dict[str, Dict[str, str]] = {
    "standard": {**_STAGE1, "peak_conv": "conv1d_6"},
    "ensemble": {
        **_STAGE1,
        "s2_enc_conv1": "conv1d_6",
        "s2_enc_conv1_bn": "batch_normalization_5",
        "s2_key_conv": "conv1d_7",
        "s2_conv2d_1": "conv2d_2",
        "s2_conv2d_1_bn": "batch_normalization_6",
        "s2_conv2d_2": "conv2d_3",
        "s2_conv2d_2_bn": "batch_normalization_7",
        "s2_dec_conv1": "conv1d_8",
        "s2_dec_conv1_bn": "batch_normalization_8",
        "s2_dec_conv2": "conv1d_9",
        "s2_dec_conv2_bn": "batch_normalization_9",
        "peak_conv": "conv1d_10",
        "noise_conv": "conv1d_11",
    },
}
