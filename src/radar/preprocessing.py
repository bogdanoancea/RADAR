# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Input normalisation and its inverse.

The networks were trained on spectra scaled to [0, 1]. Real spectra are
therefore min-max normalised per spectrum before decomposition, and the
predicted components can be mapped back to the original intensity units with
:func:`denormalize_components`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

__all__ = ["Scaling", "minmax_normalize", "denormalize_components"]


@dataclass
class Scaling:
    """Per-spectrum affine scaling ``x_norm = (x - offset) / scale``.

    Attributes
    ----------
    offset : ndarray, shape (n,)
        Per-spectrum minimum.
    scale : ndarray, shape (n,)
        Per-spectrum range (max - min); 1 for flat spectra.
    """

    offset: np.ndarray
    scale: np.ndarray

    @classmethod
    def identity(cls, n: int) -> "Scaling":
        return cls(np.zeros(n), np.ones(n))


def minmax_normalize(spectra: np.ndarray):
    """Scale each spectrum (row) to [0, 1].

    Parameters
    ----------
    spectra : ndarray, shape (n, length)

    Returns
    -------
    normalized : ndarray, shape (n, length)
    scaling : Scaling
        Parameters needed to undo the normalisation. A flat spectrum gets a
        scale of 1 (and is mapped to zeros) instead of producing NaNs.
    """
    spectra = np.asarray(spectra, dtype=np.float64)
    if spectra.ndim != 2:
        raise ValueError(f"expected a 2-D array (n, length), got shape {spectra.shape}")
    lo = spectra.min(axis=1)
    rng = spectra.max(axis=1) - lo
    rng = np.where(rng > 0, rng, 1.0)
    return (spectra - lo[:, None]) / rng[:, None], Scaling(lo, rng)


def denormalize_components(components: Dict[str, np.ndarray], scaling: Scaling) -> Dict[str, np.ndarray]:
    """Map normalised components back to the original intensity units.

    Since ``x = scale * x_norm + offset`` and ``x_norm ≈ b + cr + n + p``,
    every component is multiplied by ``scale`` and the ``offset`` is added to
    the baseline, so that the components still sum to the original spectrum.
    """
    s = scaling.scale[:, None]
    out = {k: v * s for k, v in components.items()}
    if "baseline" in out:
        out["baseline"] = out["baseline"] + scaling.offset[:, None]
    return out
