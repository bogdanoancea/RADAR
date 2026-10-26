# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Plotting helpers (require ``matplotlib``: ``pip install radar-raman[plot]``)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .models import COMPONENTS

__all__ = ["plot_components", "plot_comparison"]

_LABELS = {
    "baseline": "Baseline",
    "cosmic_rays": "Cosmic rays",
    "noise": "Noise",
    "peaks": "Peaks (corrected spectrum)",
}


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib: pip install matplotlib") from e
    return plt


def _row(a, index):
    a = np.asarray(a)
    return a if a.ndim == 1 else a[index]


def plot_components(components, index: int = 0, wavenumbers: Optional[np.ndarray] = None,
                    figsize=(9, 8), title: Optional[str] = None):
    """Plot one spectrum and its four components in five stacked panels.

    Parameters
    ----------
    components : Components
    index : int
        Spectrum to plot when ``components`` holds several.
    wavenumbers : ndarray, optional
        x axis; defaults to ``components.wavenumbers`` or the sample index.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _plt()
    x = wavenumbers if wavenumbers is not None else components.wavenumbers
    spec = _row(components.spectra, index)
    if x is None:
        x = np.arange(len(spec))
    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
    axes[0].plot(x, spec, color="black", lw=0.8, label="Input")
    axes[0].plot(x, _row(components.baseline, index), color="tab:orange", lw=1, label="Baseline")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Input")
    colors = ["tab:orange", "tab:red", "tab:gray", "tab:blue"]
    for ax, c, col in zip(axes[1:], COMPONENTS, colors):
        ax.plot(x, _row(getattr(components, c), index), color=col, lw=0.8)
        ax.set_ylabel(_LABELS[c].split(" (")[0])
    axes[-1].set_xlabel("Raman shift (cm$^{-1}$)" if components.wavenumbers is not None or wavenumbers is not None else "Index")
    fig.suptitle(title or f"RADAR decomposition ({components.model} model)")
    fig.tight_layout()
    return fig


def plot_comparison(dataset, components, index: int = 0, figsize=(9, 8)):
    """Overlay predicted and true components of a synthetic spectrum.

    Parameters
    ----------
    dataset : SyntheticDataset
        Ground truth.
    components : Components
        Prediction for (a superset of) the same spectra, in the same units.
    """
    plt = _plt()
    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
    axes[0].plot(dataset.X[index], color="black", lw=0.8)
    axes[0].set_ylabel("Input")
    for ax, c in zip(axes[1:], COMPONENTS):
        ax.plot(getattr(dataset, c)[index], color="black", lw=1, label="True")
        ax.plot(_row(getattr(components, c), index), color="tab:red", lw=0.8, ls="--", label="Predicted")
        ax.set_ylabel(_LABELS[c].split(" (")[0])
    axes[1].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Index")
    fig.tight_layout()
    return fig
