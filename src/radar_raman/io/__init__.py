# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Reading and writing spectra, decompositions and data sets in many formats.

Use :func:`read` to load spectra from any supported file and :func:`write` to
save spectra, a :class:`~radar_raman.Components` decomposition or a
:class:`~radar_raman.generator.SyntheticDataset`. The format is chosen from the file
extension (or ``format=``). :func:`formats` lists what is available.

See the user guide for the full table of formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from ._record import INTENSITY, Record, default_names
from . import _containers, _jcamp, _spc, _tabular, _vendor

__all__ = [
    "SpectralData",
    "read",
    "write",
    "formats",
    "read_spectra",
    "save_spectra",
    "save_components",
    "load_components",
    "save_dataset",
    "load_dataset",
    "Record",
]

COMPONENTS = ("baseline", "cosmic_rays", "noise", "peaks")


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------
@dataclass
class SpectralData(Record):
    """Spectra read from a file (or built from arrays).

    Attributes
    ----------
    intensities : ndarray (n, length), or list of 1-D arrays
        The spectra, one per row (a list when their lengths differ).
    wavenumbers : ndarray (length,), list of arrays, or None
        Spectral axis (Raman shift, wavelength, pixel, ...) if known.
    names : list of str or None
    positions : ndarray (n, k) or None
        Coordinates of each spectrum in a map, line scan, depth or time
        series; column names in ``position_names``.
    map_shape : tuple or None
        Grid shape of a regular map, e.g. ``(ny, nx)``.
    x_units, y_units : str
    source : str
        File the data came from.
    meta : dict
        Instrument / file metadata.
    arrays : dict
        All arrays in the file; besides ``"intensity"`` this may contain
        decomposition components when the file was written by RADAR.
    """

    source: str = ""

    @property
    def intensities(self):
        return self.first_array()

    @property
    def length(self) -> int:
        a = self.intensities
        return len(a[0]) if isinstance(a, list) else a.shape[1]

    def __len__(self) -> int:
        return self.n_spectra

    def as_map(self, array: Optional[str] = None) -> np.ndarray:
        """The spectra reshaped to ``(*map_shape, length)`` (maps only)."""
        if self.map_shape is None:
            raise ValueError("these spectra do not come from a regular map")
        a = self.arrays[array] if array else self.intensities
        return np.asarray(a).reshape(*self.map_shape, -1)

    @classmethod
    def from_record(cls, rec: Record, source: str = "") -> "SpectralData":
        return cls(rec.arrays, rec.wavenumbers, rec.names, rec.positions, rec.position_names,
                   rec.map_shape, rec.x_units, rec.y_units, rec.meta, source)

    @classmethod
    def from_arrays(cls, intensities, wavenumbers=None, names=None, positions=None,
                    position_names=None, map_shape=None, x_units="", y_units="") -> "SpectralData":
        """Build from arrays: ``intensities`` of shape ``(length,)``, ``(n, length)``
        or ``(*map_shape, length)``."""
        if isinstance(intensities, list) and intensities and np.ndim(intensities[0]) == 1 and \
                len({len(v) for v in intensities}) > 1:
            arr = [np.asarray(v, dtype=np.float64) for v in intensities]
        else:
            arr = np.asarray(intensities, dtype=np.float64)
            if arr.ndim > 2:
                map_shape = map_shape or tuple(arr.shape[:-1])
                arr = arr.reshape(-1, arr.shape[-1])
            arr = np.atleast_2d(arr)
        rec = Record({INTENSITY: arr}, wavenumbers, names, positions, list(position_names or []),
                     map_shape, x_units, y_units).normalized()
        return cls.from_record(rec)

    def __repr__(self) -> str:
        wn = ""
        if self.wavenumbers is not None and not isinstance(self.wavenumbers, list):
            wn = f", axis={self.wavenumbers[0]:g}..{self.wavenumbers[-1]:g}"
            if self.x_units:
                wn += f" {self.x_units}"
        extra = f", map_shape={self.map_shape}" if self.map_shape else ""
        L = "ragged" if self.ragged else self.length
        return f"SpectralData(n={len(self)}, length={L}{wn}{extra}, source={Path(self.source).name!r})"


# ---------------------------------------------------------------------------
# Format registry
# ---------------------------------------------------------------------------
@dataclass
class Format:
    name: str
    extensions: tuple
    description: str
    reader: Optional[Callable] = None
    writer: Optional[Callable] = None
    requires: str = ""
    multi_array: bool = True  # can hold several arrays (decomposition) in one file


def _text_reader(delim=None):
    def read(path, layout="auto", delimiter=None, decimal=None, comments="#", skip_header=0,
             encoding=None, **_):
        header, cols = _tabular.read_text_table(path, delimiter if delimiter is not None else delim,
                                                decimal, comments, skip_header, encoding)
        return _tabular.table_to_record(cols, header, layout)
    return read


def _text_writer(delim):
    def write(path, record, layout=None, decimal=".", **_):
        cols = _choose_layout(record, layout, default_wide=True)
        return _tabular.write_text_table(path, cols, delim, decimal)
    return write


def _choose_layout(record: Record, layout: Optional[str], default_wide: bool):
    if layout is None:
        layout = "wide" if (default_wide and len(record.arrays) == 1 and not record.ragged
                            and record.positions is None) else "long"
    if layout == "wide":
        if len(record.arrays) != 1:
            raise ValueError("layout='wide' holds one array; use layout='long' for decompositions")
        return _tabular.record_to_wide(record)
    if layout == "long":
        return _tabular.record_to_long(record)
    raise ValueError("layout must be 'wide' or 'long'")


def _read_excel(path, sheet=0, layout="auto", **_):
    if sheet is None or sheet == "all":
        names = [s for s in _tabular.excel_sheet_names(path) if s != "radar_metadata"]
    else:
        names = [sheet]
    recs = {}
    for s in names:
        header, cols = _tabular.read_excel_table(path, s)
        recs[str(s)] = _tabular.table_to_record(cols, header, layout)
    if len(recs) == 1:
        return next(iter(recs.values()))
    first = next(iter(recs.values()))
    first.arrays = {k: r.first_array() for k, r in recs.items()}
    return first


def _write_excel(path, record, layout=None, **_):
    if layout == "long" or record.ragged or record.positions is not None:
        sheets = {"spectra": _tabular.record_to_long(record)}
    elif len(record.arrays) == 1:
        sheets = {"spectra": _tabular.record_to_wide(record)}
    else:  # one sheet per array
        sheets = {k: _tabular.record_to_wide(r) for k, r in record.split().items()}
    return _tabular.write_excel(path, sheets, {"x_units": record.x_units, **record.meta})


def _read_parquet(path, layout="auto", **_):
    names, cols, meta = _tabular.read_parquet_table(path)
    rec = _tabular.table_to_record(cols, names, layout)
    _restore_meta(rec, meta)
    return rec


def _write_parquet(path, record, layout=None, **_):
    cols = _choose_layout(record, layout, default_wide=False)
    return _tabular.write_parquet(path, cols, _record_meta(record))


def _read_feather(path, layout="auto", **_):
    names, cols = _tabular.read_feather_table(path)
    return _tabular.table_to_record(cols, names, layout)


def _write_feather(path, record, layout=None, **_):
    return _tabular.write_feather(path, _choose_layout(record, layout, default_wide=False))


def _read_sqlite(path, table=None, query=None, layout="auto", **_):
    names, cols, meta, _tables = _tabular.read_sqlite_table(path, table, query)
    rec = _tabular.table_to_record(cols, names, layout)
    _restore_meta(rec, meta)
    return rec


def _write_sqlite(path, record, layout=None, table="spectra", if_exists="replace", **_):
    cols = _choose_layout(record, layout or "long", default_wide=False)
    return _tabular.write_sqlite(path, cols, _record_meta(record), table, if_exists)


def _record_meta(record: Record) -> dict:
    return {"x_units": record.x_units, "y_units": record.y_units,
            "map_shape": list(record.map_shape) if record.map_shape else None,
            "position_names": record.position_names, "meta": record.meta}


def _restore_meta(rec: Record, meta: dict):
    if not meta:
        return
    rec.x_units = meta.get("x_units", rec.x_units) or rec.x_units
    rec.y_units = meta.get("y_units", rec.y_units) or rec.y_units
    if meta.get("map_shape"):
        rec.map_shape = tuple(meta["map_shape"])
    if isinstance(meta.get("meta"), dict):
        rec.meta.update(meta["meta"])


def _single(fn):
    def read(path, **_):
        return fn(path)
    return read


def _keyed(fn):
    def read(path, key=None, **_):
        return fn(path, key)
    return read


def _plain(fn):
    def write(path, record, **_):
        return fn(path, record)
    return write


FORMATS: List[Format] = [
    Format("csv", (".csv",), "Comma-separated text (wide or long layout; ';' and decimal commas "
           "are detected)", _text_reader(None), _text_writer(",")),
    Format("tsv", (".tsv", ".tab"), "Tab-separated text", _text_reader("\t"), _text_writer("\t")),
    Format("text", (".txt", ".dat", ".asc", ".prn", ".xy"),
           "Delimited text and instrument text exports (Renishaw WiRE, Horiba LabSpec, WITec, "
           "Ocean Insight, Wasatch, B&W Tek, RRUFF, ...)", _text_reader(None), _text_writer("\t")),
    Format("rruff", (".rruff",), "RRUFF database text files", _text_reader(None), None),
    Format("excel", (".xlsx", ".xlsm", ".xls"), "Excel workbook", _read_excel, _write_excel,
           "pandas, openpyxl"),
    Format("parquet", (".parquet", ".pq"), "Apache Parquet (long or wide table)", _read_parquet,
           _write_parquet, "pandas, pyarrow"),
    Format("feather", (".feather", ".arrow"), "Apache Arrow / Feather", _read_feather, _write_feather,
           "pandas, pyarrow"),
    Format("sqlite", (".db", ".sqlite", ".sqlite3"), "SQLite database (long table + metadata)",
           _read_sqlite, _write_sqlite),
    Format("hdf5", (".h5", ".hdf5", ".he5", ".nc", ".nc4"), "HDF5 / NetCDF-4", _keyed(_containers.read_h5),
           _plain(_containers.write_h5)),
    Format("npz", (".npz",), "NumPy archive", _keyed(_containers.read_npz), _plain(_containers.write_npz)),
    Format("npy", (".npy",), "NumPy array", _single(_containers.read_npy), _plain(_containers.write_npy),
           multi_array=False),
    Format("matlab", (".mat",), "MATLAB v5-v7.3 (variables or structs)",
           _keyed(_containers.read_mat), _plain(_containers.write_mat)),
    Format("json", (".json",), "JSON", _keyed(_containers.read_json), _plain(_containers.write_json)),
    Format("spc", (".spc",), "Thermo Galactic GRAMS SPC (new and old format, single and multi-file)",
           _single(_spc.read_spc), _plain(_spc.write_spc), multi_array=False),
    Format("jcamp", (".jdx", ".dx", ".jcamp", ".jcm"), "JCAMP-DX 4.24 / 5.x",
           _single(_jcamp.read_jcamp), _plain(_jcamp.write_jcamp), multi_array=False),
    Format("wdf", (".wdf",), "Renishaw WiRE", _single(_vendor.read_wdf), None, "renishawWiRE"),
    Format("labspec", (".xml",), "Horiba LabSpec 6 XML export", _single(_vendor.read_labspec_xml),
           None, "rosettasciio"),
    Format("trivista", (".tvf",), "Princeton Instruments TriVista", _single(_vendor.read_trivista), None,
           "rosettasciio"),
]

_BY_NAME = {f.name: f for f in FORMATS}


def _detect(path: Path, fmt: Optional[str]) -> Format:
    if fmt is not None:
        key = fmt.lower().lstrip(".")
        if key in _BY_NAME:
            return _BY_NAME[key]
        for f in FORMATS:
            if f".{key}" in f.extensions:
                return f
        raise ValueError(f"unknown format {fmt!r}; see radar_raman.io.formats()")
    name = path.name.lower()
    if name.endswith(".xml"):
        return _BY_NAME["labspec"]
    for f in FORMATS:
        if any(name.endswith(ext) for ext in f.extensions):
            return f
    raise ValueError(f"cannot tell the format of {path.name!r} from its extension; pass format=... "
                     "(see radar_raman.io.formats())")


def formats(show: bool = False) -> List[dict]:
    """Supported formats: name, extensions, read/write support and requirements."""
    rows = []
    for f in FORMATS:
        rows.append({"format": f.name, "extensions": " ".join(f.extensions), "read": f.reader is not None,
                     "write": f.writer is not None, "requires": f.requires or "-",
                     "description": f.description})
    if show:
        for r in rows:
            rw = ("R" if r["read"] else "-") + ("W" if r["write"] else "-")
            print(f"{r['format']:9s} {rw}  {r['extensions']:32s} {r['requires']:18s} {r['description']}")
    return rows


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def _infer_map_shape(rec: Record) -> None:
    """Set ``map_shape`` when the x/y positions form a complete regular grid."""
    if rec.map_shape is not None or rec.positions is None or rec.positions.shape[1] < 2:
        return
    names = [p.lower() for p in rec.position_names]
    ix = names.index("x") if "x" in names else 0
    iy = names.index("y") if "y" in names else 1
    x, y = rec.positions[:, ix], rec.positions[:, iy]
    nx, ny = len(np.unique(x)), len(np.unique(y))
    if nx < 2 or ny < 2 or nx * ny != len(x):
        return
    # row-major: y constant along each row of nx spectra, x pattern repeated
    yy, xx = y.reshape(ny, nx), x.reshape(ny, nx)
    if np.all(yy == yy[:, :1]) and np.all(xx == xx[:1]):
        rec.map_shape = (ny, nx)
        return
    # column-major: x constant along each block of ny spectra
    xx, yy = x.reshape(nx, ny), y.reshape(nx, ny)
    if np.all(xx == xx[:, :1]) and np.all(yy == yy[:1]):
        rec.map_shape = (nx, ny)


def _finish(rec: Record, source: str) -> SpectralData:
    rec = rec.normalized()
    if rec.names is not None and len(rec.names) != rec.n_spectra:
        rec.names = None
    _infer_map_shape(rec)
    return SpectralData.from_record(rec, source)


# ---------------------------------------------------------------------------
# Public read / write
# ---------------------------------------------------------------------------
def read(path, format: Optional[str] = None, **options) -> SpectralData:
    """Read spectra from a file.

    Parameters
    ----------
    path : str or Path
    format : str, optional
        Format name or extension (``"csv"``, ``"spc"``, ``"wdf"``, ...);
        detected from the extension when omitted. See :func:`formats`.
    **options
        Format-specific options:

        * text (csv/tsv/txt...): ``layout`` (``"auto"``, ``"wide"``/``"columns"``,
          ``"rows"``, ``"long"``), ``delimiter``, ``decimal`` (``"."``/``","``),
          ``comments``, ``skip_header``, ``encoding``;
        * Excel: ``sheet`` (index, name, or ``"all"``), ``layout``;
        * Parquet / Feather: ``layout``;
        * SQLite: ``table``, ``query`` (any SQL returning a table), ``layout``;
        * HDF5 / NPZ / MAT / JSON: ``key`` (name of the array holding the spectra).

    Returns
    -------
    SpectralData
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    fmt = _detect(path, format)
    if fmt.reader is None:
        raise ValueError(f"reading {fmt.name} files is not supported")
    rec = fmt.reader(path, **options)
    out = _finish(rec, str(path))
    out.meta.setdefault("format", fmt.name)
    return out


def _to_record(obj, wavenumbers=None, names=None) -> Record:
    from ..components import Components
    from ..generator import SyntheticDataset

    if isinstance(obj, Components):
        return obj.to_record()
    if isinstance(obj, SyntheticDataset):
        arrays = {"spectra": obj.X, **{c: getattr(obj, c) for c in COMPONENTS}}
        return Record(arrays, wavenumbers, names, meta={"params": obj.params, "kind": "synthetic_dataset"})
    if isinstance(obj, Record):
        return obj
    if hasattr(obj, "intensities"):
        return Record({INTENSITY: obj.intensities}, getattr(obj, "wavenumbers", None),
                      getattr(obj, "names", None))
    data = SpectralData.from_arrays(obj, wavenumbers, names)
    return data


def write(obj, path, format: Optional[str] = None, wavenumbers=None, names=None, **options):
    """Write spectra, a decomposition or a synthetic data set to a file.

    Parameters
    ----------
    obj : SpectralData, Components, SyntheticDataset, or array-like
        What to write. Arrays are ``(length,)``, ``(n, length)`` or
        ``(*map_shape, length)``.
    path : str or Path
        Output file; the format follows the extension (or ``format=``).
    wavenumbers, names : optional
        Spectral axis and spectrum names when ``obj`` is a plain array.
    **options
        ``layout`` (``"wide"`` / ``"long"``) for tables, ``decimal`` for
        text, ``table`` / ``if_exists`` for SQLite.

    Returns
    -------
    Path or list of Path
        Formats that hold one array per file (SPC, JCAMP-DX, NPY) write a
        decomposition as one file per component, ``<stem>_<component><ext>``,
        and return the list of paths.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = _detect(path, format)
    if fmt.writer is None:
        raise ValueError(f"writing {fmt.name} files is not supported (read-only format)")
    rec = _to_record(obj, wavenumbers, names).normalized()
    if not fmt.multi_array and len(rec.arrays) > 1:
        out = []
        for k, sub in rec.split().items():
            p = path.with_name(f"{path.stem}_{k}{path.suffix}")
            fmt.writer(p, sub, **options)
            out.append(p)
        return out
    fmt.writer(path, rec, **options)
    return path


# ---------------------------------------------------------------------------
# Convenience wrappers (also used by the Components / SyntheticDataset classes)
# ---------------------------------------------------------------------------
def read_spectra(path, format: Optional[str] = None, **options) -> SpectralData:
    """Alias of :func:`read`."""
    return read(path, format, **options)


def save_spectra(path, spectra, wavenumbers=None, names: Optional[Sequence[str]] = None, **options):
    """Write spectra ``(n, length)`` to ``path`` (any writable format)."""
    return write(spectra, path, wavenumbers=wavenumbers, names=names, **options)


def save_components(components, path, **options):
    """Write a :class:`~radar_raman.Components` decomposition (see :func:`write`)."""
    return write(components, path, **options)


def load_components(path, format: Optional[str] = None, **options):
    """Read a decomposition written by :func:`save_components`.

    Works for every multi-array format (NPZ, HDF5, MAT, JSON, CSV/TSV/TXT,
    Parquet, Feather, SQLite, Excel). For one-array formats pass the file of
    any component (``<stem>_peaks.spc``); the sibling files are found
    automatically.
    """
    from ..components import Components

    path = Path(path)
    fmt = _detect(path, format)
    if not fmt.multi_array:
        stem = path.stem
        for k in ("spectra",) + COMPONENTS:
            if stem.endswith("_" + k):
                stem = stem[: -len(k) - 1]
        arrays = {}
        first = None
        for k in ("spectra",) + COMPONENTS:
            p = path.with_name(f"{stem}_{k}{path.suffix}")
            d = read(p, format, **options)
            arrays[k] = d.intensities
            first = first or d
        rec = Record(arrays, first.wavenumbers, first.names, first.positions, first.position_names,
                     first.map_shape, first.x_units, first.y_units, first.meta)
    else:
        if fmt.name == "excel":
            options.setdefault("sheet", "all")
        rec = read(path, format, **options)
    missing = [k for k in ("spectra",) + COMPONENTS if k not in rec.arrays]
    if missing:
        raise ValueError(f"{path} does not contain a decomposition (missing {missing})")
    return Components.from_record(rec)


def save_dataset(dataset, path, **options):
    """Write a :class:`~radar_raman.generator.SyntheticDataset` (any multi-array format)."""
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(".npz")
    return write(dataset, path, **options)


def load_dataset(path, format: Optional[str] = None, **options):
    """Read a data set written by :func:`save_dataset` (or by RADAR 0.1)."""
    from ..generator import SyntheticDataset

    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(".npz")
    fmt = _detect(path, format)
    if fmt.name == "excel":
        options.setdefault("sheet", "all")
    rec = read(path, format, **options)
    arrays = rec.arrays
    X = arrays.get("spectra", arrays.get(INTENSITY))
    missing = [c for c in COMPONENTS if c not in arrays]
    if X is None or missing:
        raise ValueError(f"{path} is not a RADAR data set (missing {missing or ['spectra']})")
    params = rec.meta.get("params", {})
    if not params:
        params = {k: v for k, v in rec.meta.items() if k in ("n", "length", "min_width", "max_width",
                                                             "max_num_peaks", "increase_noise",
                                                             "legacy_scaling", "seed")}
    return SyntheticDataset(np.asarray(X, dtype=np.float32),
                            *(np.asarray(arrays[c], dtype=np.float32) for c in COMPONENTS),
                            params=params)
