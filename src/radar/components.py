# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Container for the result of a decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .models import COMPONENTS

__all__ = ["Components"]


@dataclass
class Components:
    """The four components of one or more Raman spectra.

    Arrays are 1-D ``(length,)`` when a single spectrum was decomposed and
    2-D ``(n, length)`` otherwise.

    Attributes
    ----------
    spectra : ndarray
        The input spectra (in the units of :attr:`units`).
    baseline : ndarray
        Smooth background (fluorescence, etc.).
    cosmic_rays : ndarray
        Narrow spikes caused by cosmic rays.
    noise : ndarray
        Random noise.
    peaks : ndarray
        Raman bands -- the corrected, denoised spectrum.
    wavenumbers : ndarray or None
        Spectral axis, if known (e.g. read from a file).
    units : {"original", "normalized"}
        ``"original"``: same units as the input; ``"normalized"``: the
        per-spectrum min-max scale in which the network works.
    model : str
        Name of the model that produced the decomposition.
    names : list of str or None
        Spectrum names (carried over from the input file).
    positions : ndarray (n, k) or None
        Map / scan coordinates of each spectrum; names in ``position_names``.
    map_shape : tuple or None
        Grid shape of a regular map; see :meth:`as_map`.
    meta : dict
        Normalisation parameters and input-file metadata.
    """

    spectra: np.ndarray
    baseline: np.ndarray
    cosmic_rays: np.ndarray
    noise: np.ndarray
    peaks: np.ndarray
    wavenumbers: Optional[np.ndarray] = None
    units: str = "original"
    model: str = ""
    meta: dict = field(default_factory=dict)
    names: Optional[List[str]] = None
    positions: Optional[np.ndarray] = None
    position_names: List[str] = field(default_factory=list)
    map_shape: Optional[tuple] = None
    x_units: str = ""

    # ------------------------------------------------------------------
    @property
    def n_spectra(self) -> int:
        return 1 if self.spectra.ndim == 1 else self.spectra.shape[0]

    @property
    def length(self) -> int:
        return self.spectra.shape[-1]

    @property
    def corrected(self) -> np.ndarray:
        """Spectrum with baseline, cosmic rays and noise removed (= :attr:`peaks`)."""
        return self.peaks

    @property
    def denoised(self) -> np.ndarray:
        """Input minus the estimated noise (baseline and peaks kept)."""
        return self.spectra - self.noise

    @property
    def despiked(self) -> np.ndarray:
        """Input minus the estimated cosmic rays."""
        return self.spectra - self.cosmic_rays

    @property
    def baseline_corrected(self) -> np.ndarray:
        """Input minus the estimated baseline (noise and peaks kept)."""
        return self.spectra - self.baseline

    @property
    def reconstruction(self) -> np.ndarray:
        """Sum of the four components."""
        return self.baseline + self.cosmic_rays + self.noise + self.peaks

    @property
    def residual(self) -> np.ndarray:
        """``spectra - reconstruction``.

        Exactly zero (up to rounding) for the standard model, whose noise is
        defined as the remainder; small but non-zero for the ensemble model,
        which predicts the noise directly.
        """
        return self.spectra - self.reconstruction

    def as_map(self, component: str = "peaks") -> np.ndarray:
        """A component (or ``"spectra"``) reshaped to ``(*map_shape, length)``.

        Only for spectra read from a regular map (Renishaw / LabSpec /
        TriVista maps, long tables with x, y columns, ...).
        """
        if self.map_shape is None:
            raise ValueError("these spectra do not come from a regular map")
        return np.asarray(getattr(self, component)).reshape(*self.map_shape, -1)

    def band_image(self, lo: float, hi: float, component: str = "peaks") -> np.ndarray:
        """Integrated intensity of ``component`` between two wavenumbers, as a
        map image of shape ``map_shape`` (or a vector of length ``n``)."""
        if self.wavenumbers is None:
            raise ValueError("wavenumbers are needed to select a band")
        wn = np.asarray(self.wavenumbers)
        sel = (wn >= min(lo, hi)) & (wn <= max(lo, hi))
        if not np.any(sel):
            raise ValueError(f"no points between {lo} and {hi}")
        vals = np.atleast_2d(getattr(self, component))[:, sel]
        x = wn[sel]
        order = np.argsort(x)
        area = np.trapezoid(vals[:, order], x[order], axis=1) if hasattr(np, "trapezoid") else \
            np.trapz(vals[:, order], x[order], axis=1)
        return area.reshape(self.map_shape) if self.map_shape is not None else area

    # ------------------------------------------------------------------
    def to_dict(self, include_input: bool = True) -> Dict[str, np.ndarray]:
        """Arrays as a dict ``{"spectra", "baseline", ..., "wavenumbers"}``."""
        d = {c: getattr(self, c) for c in COMPONENTS}
        if include_input:
            d = {"spectra": self.spectra, **d}
        if self.wavenumbers is not None:
            d["wavenumbers"] = self.wavenumbers
        return d

    def to_record(self):
        """Format-independent representation used by :mod:`radar.io`."""
        from .io import Record

        arrays = {k: np.atleast_2d(getattr(self, k)) for k in ("spectra",) + COMPONENTS}
        meta = {"units": self.units, "model": self.model, "kind": "decomposition"}
        for k in ("offset", "scale"):
            if k in self.meta:
                meta[k] = np.atleast_1d(self.meta[k]).tolist()
        return Record(arrays, self.wavenumbers, self.names, self.positions, list(self.position_names),
                      self.map_shape, self.x_units, "", meta)

    @classmethod
    def from_record(cls, rec) -> "Components":
        meta = dict(rec.meta)
        arrays = {k: np.asarray(rec.arrays[k]) for k in ("spectra",) + COMPONENTS}
        return cls(**arrays, wavenumbers=rec.wavenumbers, units=str(meta.get("units", "original")),
                   model=str(meta.get("model", "")), meta=meta, names=rec.names,
                   positions=rec.positions, position_names=list(rec.position_names),
                   map_shape=rec.map_shape, x_units=rec.x_units)

    def __getitem__(self, index) -> "Components":
        """Select spectra, e.g. ``comps[0]`` or ``comps[10:20]``."""
        if self.spectra.ndim == 1:
            raise IndexError("Components holds a single spectrum")
        names = None
        if self.names is not None:
            names = list(np.asarray(self.names, dtype=object)[index]) if not isinstance(index, int) \
                else [self.names[index]]
        return Components(
            *(getattr(self, k)[index] for k in ("spectra",) + COMPONENTS),
            wavenumbers=self.wavenumbers,
            units=self.units,
            model=self.model,
            meta=dict(self.meta),
            names=names,
            positions=None if self.positions is None else np.atleast_2d(self.positions[index]),
            position_names=list(self.position_names),
            map_shape=None,
            x_units=self.x_units,
        )

    def __len__(self) -> int:
        return self.n_spectra

    def save(self, path, **options):
        """Save in any format supported by :func:`radar.write` (chosen by extension)."""
        from .io import write

        return write(self, path, **options)

    def plot(self, index: int = 0, **kwargs):
        """Plot the decomposition of one spectrum (see :func:`radar.plotting.plot_components`)."""
        from .plotting import plot_components

        return plot_components(self, index=index, **kwargs)

    def __repr__(self) -> str:
        extra = f", map_shape={self.map_shape}" if self.map_shape else ""
        return (
            f"Components(n_spectra={self.n_spectra}, length={self.length}, "
            f"units={self.units!r}, model={self.model!r}{extra})"
        )
