# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Proprietary instrument formats read through optional third-party packages.

=====================  ===========  ==========================================
Instrument / software  Extension    Package (install separately)
=====================  ===========  ==========================================
Renishaw WiRE          ``.wdf``     ``renishawWiRE`` (MIT); falls back to
                                    ``rosettasciio``
Horiba LabSpec 6       ``.xml``     ``rosettasciio`` (GPL-3.0)
Princeton TriVista     ``.tvf``     ``rosettasciio`` (GPL-3.0)
=====================  ===========  ==========================================

The packages are imported only when such a file is read; RADAR itself does
not depend on (or include) them.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from ._record import INTENSITY, Record


def _missing(pkg: str, fmt: str) -> ImportError:
    return ImportError(f"Reading {fmt} files needs the optional package {pkg!r}: pip install {pkg}")


def read_wdf(path) -> Record:
    """Renishaw WiRE ``.wdf`` (single spectra, series, line scans and maps)."""
    try:
        from renishawWiRE import WDFReader
    except ImportError:
        try:
            return read_rsciio(path, "renishaw")
        except ImportError:
            raise _missing("renishawWiRE", "Renishaw .wdf") from None
    import contextlib
    import io as _io

    with contextlib.redirect_stdout(_io.StringIO()):
        r = WDFReader(str(path))
    spectra = np.asarray(r.spectra, dtype=np.float64)
    x = np.asarray(r.xdata, dtype=np.float64)
    map_shape = None
    if spectra.ndim == 3:
        map_shape = tuple(spectra.shape[:2])
        spectra = spectra.reshape(-1, spectra.shape[-1])
    spectra = np.atleast_2d(spectra)
    n = spectra.shape[0]
    pos_cols, pos_names = [], []
    for name in ("xpos", "ypos", "zpos"):
        v = getattr(r, name, None)
        if v is not None and np.size(v) == n:
            v = np.asarray(v, dtype=np.float64).ravel()
            if n > 1 and np.ptp(v) == 0:
                continue
            pos_cols.append(v)
            pos_names.append(name[0])
    positions = np.column_stack(pos_cols) if pos_cols and n > 1 else None
    meta = {"format": "Renishaw WDF", "title": str(getattr(r, "title", "")),
            "measurement_type": str(getattr(r, "measurement_type", "")),
            "laser_wavenumber": float(getattr(r, "laser_length", np.nan) or np.nan)}
    return Record({INTENSITY: spectra}, x, None, positions, pos_names if positions is not None else [],
                  map_shape, str(getattr(r, "xlist_unit", "")), str(getattr(r, "spectral_unit", "")),
                  meta)


def read_rsciio(path, plugin: str) -> Record:
    """Read a file with a ``rosettasciio`` plugin (``renishaw``, ``jobinyvon``, ``trivista``)."""
    try:
        import importlib

        mod = importlib.import_module(f"rsciio.{plugin}")
    except ImportError:
        raise _missing("rosettasciio", f"{plugin}") from None
    logging.getLogger("rsciio").setLevel(logging.ERROR)
    logging.getLogger("RosettaSciIO").setLevel(logging.ERROR)
    out = mod.file_reader(str(path))
    # keep the first signal with a spectral (non-navigation) axis
    sig = next((o for o in out if any(a.get("navigate") is False for a in o["axes"])), out[0])
    data = np.asarray(sig["data"], dtype=np.float64)
    axes = sig["axes"]
    sig_axes = [a for a in axes if a.get("navigate") is False] or [axes[-1]]
    nav_axes = [a for a in axes if a.get("navigate") is True]
    sax = sig_axes[-1]
    L = data.shape[-1]
    if "axis" in sax and sax["axis"] is not None:
        x = np.asarray(sax["axis"], dtype=np.float64)
    else:
        x = sax.get("offset", 0.0) + sax.get("scale", 1.0) * np.arange(L)
    nav_shape = data.shape[:-1]
    spectra = data.reshape(-1, L)
    positions, pos_names, map_shape = None, [], None
    if nav_axes and len(nav_shape) >= 1 and spectra.shape[0] > 1:
        grids = []
        for a, size in zip(nav_axes, nav_shape):
            if "axis" in a and a["axis"] is not None:
                grids.append(np.asarray(a["axis"], dtype=np.float64))
            else:
                grids.append(a.get("offset", 0.0) + a.get("scale", 1.0) * np.arange(size))
        mesh = np.meshgrid(*grids, indexing="ij")
        positions = np.column_stack([m.ravel() for m in mesh])
        pos_names = [str(a.get("name", f"axis{i}")).lower() for i, a in enumerate(nav_axes)]
        if len(nav_shape) >= 2:
            map_shape = tuple(nav_shape)
    meta = {"format": plugin, "original_metadata_keys": list(sig.get("original_metadata", {}))[:20]}
    return Record({INTENSITY: spectra}, x, None, positions, pos_names, map_shape,
                  str(sax.get("units", "")), "", meta)


def read_labspec_xml(path) -> Record:
    return read_rsciio(path, "jobinyvon")


def read_trivista(path) -> Record:
    return read_rsciio(path, "trivista")


def available() -> List[str]:
    """Names of the optional vendor packages that are installed."""
    found = []
    for mod in ("renishawWiRE", "rsciio"):
        try:
            __import__(mod)
            found.append(mod)
        except ImportError:
            pass
    return found
