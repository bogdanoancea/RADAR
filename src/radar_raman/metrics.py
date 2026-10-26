# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""NumPy evaluation metrics for decompositions with known ground truth.

Per-spectrum metrics
--------------------
:func:`rmse`, :func:`mae`, :func:`max_error` and :func:`r2` compute one value
per spectrum (along the last axis).

Summaries
---------
:func:`evaluate_components` gives one number per metric and component.
:func:`error_statistics` gives the mean, median and variance over spectra of
the per-spectrum RMSE and max error -- the summary used in the original
evaluation notebooks.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .models import COMPONENTS

__all__ = [
    "rmse",
    "mae",
    "max_error",
    "r2",
    "per_spectrum_errors",
    "error_statistics",
    "evaluate_components",
    "snr",
]


def rmse(y_true, y_pred, axis=-1):
    """Root mean squared error along ``axis`` (per spectrum by default)."""
    return np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2, axis=axis))


def mae(y_true, y_pred, axis=-1):
    """Mean absolute error along ``axis``."""
    return np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true)), axis=axis)


def max_error(y_true, y_pred, axis=-1):
    """Maximum absolute error along ``axis``: ``max |y_true - y_pred|``.

    Per spectrum this is the worst point-wise deviation of the estimated
    component, the same quantity as ``sklearn.metrics.max_error`` applied to
    one spectrum. Unlike RMSE it is sensitive to a single badly estimated
    point, e.g. a missed cosmic ray or a clipped peak top.
    """
    return np.max(np.abs(np.asarray(y_pred) - np.asarray(y_true)), axis=axis)


def r2(y_true, y_pred, axis=-1, eps=1e-12):
    """Coefficient of determination along ``axis``.

    Undefined for (near-)constant targets, e.g. a spectrum without cosmic
    rays; such spectra get ``nan``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=axis)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=axis, keepdims=True)) ** 2, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ss_tot > eps, 1 - ss_res / ss_tot, np.nan)


def _get(obj, c):
    return obj[c] if isinstance(obj, dict) else getattr(obj, c)


def per_spectrum_errors(truth, prediction) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-spectrum RMSE, MAE and max error for every component.

    Parameters
    ----------
    truth, prediction
        Objects with ``baseline``, ``cosmic_rays``, ``noise`` and ``peaks``
        attributes (e.g. :class:`~radar_raman.generator.SyntheticDataset` and
        :class:`~radar_raman.Components`) or dicts with those keys, in the same units.

    Returns
    -------
    dict
        ``{component: {"rmse": (n,), "mae": (n,), "max_error": (n,)}}``.
    """
    out = {}
    for c in COMPONENTS:
        t, p = np.atleast_2d(_get(truth, c)), np.atleast_2d(_get(prediction, c))
        out[c] = {"rmse": rmse(t, p), "mae": mae(t, p), "max_error": max_error(t, p)}
    return out


def error_statistics(truth, prediction) -> Dict[str, Dict[str, float]]:
    """Mean, median and variance over spectra of the per-spectrum RMSE and max error.

    This is the summary table of the original evaluation notebooks
    (``RMSE:mean``, ``RMSE:median``, ``RMSE:variance``, ``MaxError:mean``,
    ``MaxError:median``, ``MaxError:variance``).

    Returns
    -------
    dict
        ``{component: {"rmse_mean", "rmse_median", "rmse_variance",
        "max_error_mean", "max_error_median", "max_error_variance"}}``.
    """
    out = {}
    for c, errs in per_spectrum_errors(truth, prediction).items():
        d = {}
        for name in ("rmse", "max_error"):
            v = errs[name]
            d[f"{name}_mean"] = float(np.mean(v))
            d[f"{name}_median"] = float(np.median(v))
            d[f"{name}_variance"] = float(np.var(v))
        out[c] = d
    return out


def evaluate_components(truth, prediction) -> Dict[str, Dict[str, float]]:
    """Per-component error summary.

    Parameters
    ----------
    truth, prediction
        Objects with ``baseline``, ``cosmic_rays``, ``noise`` and ``peaks``
        attributes (e.g. :class:`~radar_raman.generator.SyntheticDataset` and
        :class:`~radar_raman.Components`) or dicts with those keys, in the same units.

    Returns
    -------
    dict
        ``{component: {"rmse", "mae", "max_error", "r2"}}``. RMSE, MAE and max
        error are computed per spectrum and averaged over spectra. R^2 is
        pooled over all points of all spectra (as the ``r_square`` training
        metric), because the per-spectrum R^2 is undefined or unstable for
        spectra whose component is (nearly) constant, e.g. a spectrum without
        cosmic rays. For medians and variances use :func:`error_statistics`.
    """
    errs = per_spectrum_errors(truth, prediction)
    out = {}
    for c in COMPONENTS:
        t, p = np.atleast_2d(_get(truth, c)), np.atleast_2d(_get(prediction, c))
        out[c] = {
            "rmse": float(np.mean(errs[c]["rmse"])),
            "mae": float(np.mean(errs[c]["mae"])),
            "max_error": float(np.mean(errs[c]["max_error"])),
            "r2": float(r2(t.ravel(), p.ravel())),
        }
    return out


def snr(signal, noise, predicted_noise=0.0, eps=1e-4):
    """Signal-to-noise ratio ``mean(signal) / std(noise - predicted_noise)``.

    The definition used in the original evaluation notebooks (after
    Han et al.): ``signal`` are intensities in a band region, ``noise`` the
    intensities in a band-free region after removing a linear trend.
    """
    return float(np.mean(signal) / (np.std(np.asarray(noise) - predicted_noise) + eps))
