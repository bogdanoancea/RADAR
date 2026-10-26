# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Array container formats: NumPy (``.npy``, ``.npz``), HDF5 / NetCDF-4
(``.h5``, ``.hdf5``, ``.nc``), MATLAB (``.mat``) and JSON (``.json``).

Files written by RADAR store every array of the record plus the spectral
axis, names, positions and metadata, and are read back losslessly. Files
from other software are interpreted heuristically: the largest numeric array
is taken as the spectra and a 1-D array matching its length as the axis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ._record import INTENSITY, Record

_DATA_KEYS = ("intensity", "intensities", "spectra", "spectrum", "x_data", "X", "data", "y",
              "counts", "signal")
_AXIS_KEYS = ("wavenumbers", "wavenumber", "raman_shift", "ramanshift", "shift", "axis",
              "axisscale", "x", "xdata", "wavelength", "wavelengths", "spectral_axis")
COMPONENT_KEYS = ("spectra", "baseline", "cosmic_rays", "noise", "peaks")


def _meta_json(record: Record) -> str:
    return json.dumps({
        "radar_format": 2, "x_units": record.x_units, "y_units": record.y_units,
        "position_names": record.position_names,
        "map_shape": list(record.map_shape) if record.map_shape else None,
        "arrays": list(record.arrays), "meta": record.meta,
    }, default=str)


def _apply_meta(record: Record, js: Optional[str]) -> Record:
    if not js:
        return record
    try:
        m = json.loads(js)
    except (TypeError, ValueError):
        return record
    record.x_units = m.get("x_units", "") or ""
    record.y_units = m.get("y_units", "") or ""
    record.position_names = list(m.get("position_names") or [])
    record.map_shape = tuple(m["map_shape"]) if m.get("map_shape") else None
    record.meta.update(m.get("meta") or {})
    return record


def _ragged_pack(a):
    """Lists of 1-D arrays are stored NaN-padded plus a lengths vector."""
    if not isinstance(a, list):
        return np.asarray(a), None
    lengths = np.array([len(v) for v in a])
    out = np.full((len(a), lengths.max()), np.nan)
    for i, v in enumerate(a):
        out[i, : len(v)] = v
    return out, lengths


def _ragged_unpack(a, lengths):
    if lengths is None:
        return a
    return [np.asarray(a[i, : int(L)]) for i, L in enumerate(lengths)]


# ---------------------------------------------------------------------------
# heuristic interpretation of foreign files
# ---------------------------------------------------------------------------
def _flatten(obj, prefix="") -> Dict[str, np.ndarray]:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).startswith("__"):
                continue
            out.update(_flatten(v, f"{prefix}{k}/"))
    elif isinstance(obj, (list, tuple)) and obj and not np.isscalar(obj[0]):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}{i}/"))
    else:
        try:
            arr = np.asarray(obj)
        except Exception:
            return out
        if arr.dtype.kind in "biuf" and arr.size > 1:
            out[prefix.rstrip("/")] = arr
    return out


def interpret_arrays(arrays: Dict[str, np.ndarray], key: Optional[str] = None) -> Record:
    """Pick spectra and axis from a dict of arrays of unknown layout."""
    if not arrays:
        raise ValueError("no numeric arrays found")
    leaf = {k.split("/")[-1]: k for k in arrays}
    comp = [c for c in COMPONENT_KEYS if c in leaf]
    axis_key = next((leaf[k] for k in _AXIS_KEYS if k in leaf and np.squeeze(arrays[leaf[k]]).ndim == 1), None)
    if key is not None:
        data_key = key if key in arrays else leaf.get(key)
        if data_key is None:
            raise KeyError(f"no array named {key!r}; available: {sorted(arrays)}")
        data_keys = {INTENSITY: data_key}
    elif len(comp) >= 2:
        data_keys = {c: leaf[c] for c in comp}
    else:
        cand = next((leaf[k] for k in _DATA_KEYS if k in leaf and leaf[k] != axis_key), None)
        if cand is None:
            others = [k for k in arrays if k != axis_key]
            cand = max(others or list(arrays), key=lambda k: arrays[k].size)
        data_keys = {INTENSITY: cand}
    out = {}
    map_shape = None
    wn = None if axis_key is None else np.asarray(arrays[axis_key], dtype=np.float64).ravel()
    for name, k in data_keys.items():
        a = np.asarray(arrays[k], dtype=np.float64)
        a = np.squeeze(a) if a.ndim > 2 else a
        if a.ndim == 1:
            a = a[None, :]
        if wn is not None and a.shape[-1] != len(wn) and len(wn) in a.shape:
            ax = list(a.shape).index(len(wn))
            a = np.moveaxis(a, ax, -1)
        elif wn is None and a.ndim == 2 and a.shape[0] > a.shape[1] * 4:
            a = a.T  # long column vectors -> spectra in rows
        if a.ndim > 2:
            map_shape = tuple(a.shape[:-1])
            a = a.reshape(-1, a.shape[-1])
        out[name] = a
    if wn is not None and wn.size != next(iter(out.values())).shape[-1]:
        wn = None
    return Record(out, wn, None, None, [], map_shape)


# ---------------------------------------------------------------------------
# NumPy
# ---------------------------------------------------------------------------
def read_npy(path) -> Record:
    a = np.load(path, allow_pickle=False)
    return interpret_arrays({"data": a})


def write_npy(path, record: Record) -> Path:
    if len(record.arrays) != 1 or record.ragged:
        raise ValueError(".npy holds one rectangular array; use .npz for more")
    np.save(path, np.asarray(next(iter(record.arrays.values()))))
    return Path(path)


def read_npz(path, key: Optional[str] = None) -> Record:
    with np.load(path, allow_pickle=False) as z:
        files = {k: z[k] for k in z.files}
    if "_radar" in files:
        m = json.loads(str(files["_radar"]))
        arrays = {k: _ragged_unpack(files[f"array_{k}"], files.get(f"lengths_{k}")) for k in m["arrays"]}
        wn = files.get("wavenumbers")
        if "wavenumber_lengths" in files:
            wn = _ragged_unpack(wn, files["wavenumber_lengths"])
        names = [str(s) for s in files["names"]] if "names" in files else None
        rec = Record(arrays, wn, names, files.get("positions"))
        rec = _apply_meta(rec, str(files["_radar"]))
        if key is not None:
            rec.arrays = {INTENSITY: rec.arrays[key]}
        return rec
    # legacy RADAR 0.1 files and foreign files
    meta = None
    for mk in ("_meta", "_params"):
        if mk in files:
            meta = files.pop(mk)
    if "X" in files and all(c in files for c in COMPONENT_KEYS[1:]):
        files["spectra"] = files.pop("X")
    rec = interpret_arrays({k: v for k, v in files.items() if v.dtype.kind in "biuf"}, key)
    if meta is not None:
        try:
            rec.meta.update(json.loads(str(meta)))
        except ValueError:
            pass
    return rec


def write_npz(path, record: Record) -> Path:
    d = {"_radar": np.array(_meta_json(record))}
    for k, a in record.arrays.items():
        packed, lengths = _ragged_pack(a)
        d[f"array_{k}"] = packed
        if lengths is not None:
            d[f"lengths_{k}"] = lengths
    if record.wavenumbers is not None:
        packed, lengths = _ragged_pack(record.wavenumbers)
        d["wavenumbers"] = packed
        if lengths is not None:
            d["wavenumber_lengths"] = lengths
    if record.names:
        d["names"] = np.array([str(s) for s in record.names])
    if record.positions is not None:
        d["positions"] = record.positions
    np.savez_compressed(path, **d)
    return Path(path)


# ---------------------------------------------------------------------------
# HDF5 / NetCDF-4
# ---------------------------------------------------------------------------
def read_h5(path, key: Optional[str] = None) -> Record:
    import h5py

    with h5py.File(path, "r") as f:
        if "radar_format" in f.attrs or "_radar" in f.attrs:
            js = f.attrs.get("_radar")
            js = js.decode() if isinstance(js, bytes) else js
            names_arr = list(json.loads(js)["arrays"]) if js else [
                k for k in f.keys() if k in COMPONENT_KEYS or k == INTENSITY]
            arrays = {}
            for k in names_arr:
                a = np.asarray(f[k])
                lengths = np.asarray(f[f"{k}_lengths"]) if f"{k}_lengths" in f else None
                arrays[k] = _ragged_unpack(a, lengths)
            wn = np.asarray(f["wavenumbers"]) if "wavenumbers" in f else None
            if wn is not None and "wavenumber_lengths" in f:
                wn = _ragged_unpack(wn, np.asarray(f["wavenumber_lengths"]))
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["names"][()]] \
                if "names" in f else None
            pos = np.asarray(f["positions"]) if "positions" in f else None
            rec = Record(arrays, wn, names, pos)
            rec = _apply_meta(rec, js)
            if key is not None:
                rec.arrays = {INTENSITY: rec.arrays[key]}
            return rec
        if key is None and "Regions" in f:  # live-cell map files of the original project
            parts = [np.asarray(f["Regions"][r]["Dataset"]) for r in f["Regions"]]
            length = parts[0].shape[-1]
            data = np.concatenate([p.reshape(-1, length) for p in parts], axis=0)
            return Record({INTENSITY: data.astype(np.float64)})
        arrays = {}

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "biuf" and obj.size > 1:
                arrays[name] = np.asarray(obj)

        f.visititems(visit)
    rec = interpret_arrays(arrays, key)
    return rec


def write_h5(path, record: Record) -> Path:
    import h5py

    with h5py.File(path, "w") as f:
        for k, a in record.arrays.items():
            packed, lengths = _ragged_pack(a)
            f.create_dataset(k, data=packed, compression="gzip")
            if lengths is not None:
                f.create_dataset(f"{k}_lengths", data=lengths)
        if record.wavenumbers is not None:
            packed, lengths = _ragged_pack(record.wavenumbers)
            f.create_dataset("wavenumbers", data=packed)
            if lengths is not None:
                f.create_dataset("wavenumber_lengths", data=lengths)
        if record.names:
            f.create_dataset("names", data=np.array([str(s) for s in record.names], dtype="S"))
        if record.positions is not None:
            f.create_dataset("positions", data=record.positions)
        f.attrs["radar_format"] = 2
        f.attrs["_radar"] = _meta_json(record)
        if record.x_units:
            f.attrs["x_units"] = record.x_units
    return Path(path)


# ---------------------------------------------------------------------------
# MATLAB
# ---------------------------------------------------------------------------
def read_mat(path, key: Optional[str] = None) -> Record:
    import scipy.io

    try:
        m = scipy.io.loadmat(path, simplify_cells=True)
    except NotImplementedError:  # MATLAB v7.3 = HDF5
        return read_h5(path, key)
    if "radar_info" in m:
        js = str(m["radar_info"])
        names_arr = json.loads(js)["arrays"]
        arrays = {k: np.atleast_2d(np.asarray(m[k], dtype=np.float64)) for k in names_arr}
        wn = np.asarray(m["wavenumbers"], dtype=np.float64).ravel() if "wavenumbers" in m else None
        names = list(np.atleast_1d(m["names"])) if "names" in m else None
        pos = np.atleast_2d(m["positions"]) if "positions" in m else None
        rec = Record(arrays, wn, [str(s).strip() for s in names] if names else None, pos)
        return _apply_meta(rec, js)
    return interpret_arrays(_flatten(m), key)


def write_mat(path, record: Record) -> Path:
    import scipy.io

    if record.ragged:
        raise ValueError(".mat export of spectra with different lengths is not supported; use .h5 or .npz")
    d = {k: np.asarray(a) for k, a in record.arrays.items()}
    if record.wavenumbers is not None:
        d["wavenumbers"] = np.asarray(record.wavenumbers)
    if record.names:
        d["names"] = np.array(record.names, dtype=object)
    if record.positions is not None:
        d["positions"] = record.positions
    d["radar_info"] = _meta_json(record)  # MATLAB ignores names starting with "_"
    scipy.io.savemat(path, d, do_compression=True)
    return Path(path)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def _tolist(a):
    if a is None:
        return None
    if isinstance(a, list):
        return [np.asarray(v).tolist() for v in a]
    return np.asarray(a).tolist()


def read_json(path, key: Optional[str] = None) -> Record:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, dict) and obj.get("radar_format"):
        arrays = {}
        for k, v in obj["arrays"].items():
            lens = {len(r) for r in v}
            arrays[k] = np.asarray(v, dtype=np.float64) if len(lens) == 1 else \
                [np.asarray(r, dtype=np.float64) for r in v]
        wn = obj.get("wavenumbers")
        if wn is not None:
            wn = [np.asarray(w) for w in wn] if wn and isinstance(wn[0], list) else np.asarray(wn)
        rec = Record(arrays, wn, obj.get("names"),
                     None if obj.get("positions") is None else np.asarray(obj["positions"], dtype=np.float64),
                     list(obj.get("position_names") or []),
                     tuple(obj["map_shape"]) if obj.get("map_shape") else None,
                     obj.get("x_units", ""), obj.get("y_units", ""), obj.get("meta") or {})
        if key is not None:
            rec.arrays = {INTENSITY: rec.arrays[key]}
        return rec
    # simple foreign layouts
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):  # [{"x": [...], "y": [...]}, ...]
        xs, ys, names = [], [], []
        for i, item in enumerate(obj):
            xk = next((k for k in item if k.lower() in _AXIS_KEYS), None)
            yk = next((k for k in item if k.lower() in _DATA_KEYS or k.lower() == "y"), None)
            ys.append(np.asarray(item[yk], dtype=np.float64))
            xs.append(np.asarray(item[xk], dtype=np.float64) if xk else np.arange(len(ys[-1]), dtype=float))
            names.append(str(item.get("name", item.get("title", f"spectrum_{i + 1}"))))
        same = all(len(x) == len(xs[0]) and np.allclose(x, xs[0]) for x in xs)
        return Record({INTENSITY: np.vstack(ys) if same else ys}, xs[0] if same else xs, names)
    if isinstance(obj, dict):
        flat = {k: np.asarray(v, dtype=np.float64) for k, v in obj.items()
                if isinstance(v, list) and v and np.asarray(v).dtype.kind in "biuf"}
        return interpret_arrays(flat, key)
    return interpret_arrays({"data": np.asarray(obj, dtype=np.float64)}, key)


def write_json(path, record: Record) -> Path:
    obj = {
        "radar_format": 2,
        "x_units": record.x_units, "y_units": record.y_units,
        "wavenumbers": _tolist(record.wavenumbers),
        "names": record.names,
        "positions": _tolist(record.positions),
        "position_names": record.position_names,
        "map_shape": list(record.map_shape) if record.map_shape else None,
        "arrays": {k: _tolist(a) for k, a in record.arrays.items()},
        "meta": record.meta,
    }
    Path(path).write_text(json.dumps(obj, default=str), encoding="utf-8")
    return Path(path)
