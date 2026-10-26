# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Format-independent in-memory representation used by all readers/writers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

ArrayLike2D = Union[np.ndarray, List[np.ndarray]]

#: Name of the array holding plain spectra (no decomposition).
INTENSITY = "intensity"


@dataclass
class Record:
    """One or more named arrays of spectra sharing an axis and metadata.

    Every reader returns a ``Record`` and every writer accepts one.

    Attributes
    ----------
    arrays : dict
        ``{name: (n, length) array}``. Plain spectra use the single key
        ``"intensity"``; a decomposition uses ``"spectra"``, ``"baseline"``,
        ``"cosmic_rays"``, ``"noise"`` and ``"peaks"``. Spectra of different
        lengths are stored as a list of 1-D arrays (``ragged``).
    wavenumbers : (length,) array, list of arrays (ragged) or None
    names : list of str or None
        One name per spectrum.
    positions : (n, k) array or None
        Spatial / temporal coordinates of each spectrum (maps, line scans,
        depth profiles, time series).
    position_names : list of str
        Names of the ``k`` position columns, e.g. ``["x", "y"]``.
    map_shape : tuple or None
        Shape of the measurement grid (e.g. ``(ny, nx)``) when the spectra come
        from a regular map; ``n == prod(map_shape)``.
    x_units, y_units : str
        Units of the spectral axis and of the intensities, if known.
    meta : dict
        Any other metadata (JSON-serialisable values preferred).
    """

    arrays: Dict[str, ArrayLike2D]
    wavenumbers: Optional[ArrayLike2D] = None
    names: Optional[List[str]] = None
    positions: Optional[np.ndarray] = None
    position_names: List[str] = field(default_factory=list)
    map_shape: Optional[tuple] = None
    x_units: str = ""
    y_units: str = ""
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def ragged(self) -> bool:
        return any(isinstance(a, list) for a in self.arrays.values())

    @property
    def n_spectra(self) -> int:
        a = next(iter(self.arrays.values()))
        return len(a)

    def first_array(self) -> ArrayLike2D:
        for key in (INTENSITY, "spectra"):
            if key in self.arrays:
                return self.arrays[key]
        return next(iter(self.arrays.values()))

    def normalized(self) -> "Record":
        """Return a copy with 2-D float arrays (or lists of 1-D arrays) and
        consistent axis/names/positions."""
        arrays = {}
        for k, a in self.arrays.items():
            if isinstance(a, list):
                arrays[k] = [np.asarray(v, dtype=np.float64).ravel() for v in a]
            else:
                a = np.asarray(a, dtype=np.float64)
                arrays[k] = np.atleast_2d(a) if a.ndim <= 2 else a.reshape(-1, a.shape[-1])
        wn = self.wavenumbers
        if wn is not None and not isinstance(wn, list):
            wn = np.asarray(wn, dtype=np.float64).ravel()
        pos = None if self.positions is None else np.atleast_2d(np.asarray(self.positions, dtype=np.float64))
        if pos is not None and pos.shape[0] == 1 and self.n_spectra_from(arrays) > 1:
            pos = pos.T
        return Record(arrays, wn, None if self.names is None else [str(x) for x in self.names],
                      pos, list(self.position_names), self.map_shape, self.x_units, self.y_units,
                      dict(self.meta))

    @staticmethod
    def n_spectra_from(arrays) -> int:
        a = next(iter(arrays.values()))
        return len(a)

    def split(self) -> Dict[str, "Record"]:
        """One single-array record per array (for single-array file formats)."""
        return {
            k: Record({INTENSITY: a}, self.wavenumbers, self.names, self.positions,
                      self.position_names, self.map_shape, self.x_units, self.y_units,
                      {**self.meta, "array": k})
            for k, a in self.arrays.items()
        }

    def common_axis(self) -> Optional[np.ndarray]:
        """The shared spectral axis, or ``None`` (missing or ragged)."""
        if self.wavenumbers is None or isinstance(self.wavenumbers, list):
            return None
        return np.asarray(self.wavenumbers)

    def axis_or_index(self) -> np.ndarray:
        """The spectral axis, or ``0..length-1`` when there is none."""
        ax = self.common_axis()
        if ax is not None:
            return ax
        return np.arange(np.asarray(self.first_array()).shape[-1], dtype=np.float64)


def default_names(n: int) -> List[str]:
    return [f"spectrum_{i + 1}" for i in range(n)]


def is_monotonic(x: Sequence[float]) -> bool:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3 or not np.all(np.isfinite(x)):
        return False
    d = np.diff(x)
    return bool(np.all(d > 0) or np.all(d < 0))
