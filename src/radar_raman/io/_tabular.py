# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Tables of spectra: delimited text, Excel, Parquet, Feather and SQLite.

All tabular formats go through the same interpretation step
(:func:`table_to_record`), which recognises three layouts:

``wide`` (one spectrum per column)
    ``wavenumber, spectrum_1, spectrum_2, ...`` -- typical instrument
    exports, including two-column *x, y* files.
``rows`` (one spectrum per row)
    an optional first row with the wavenumbers, optionally preceded by empty
    cells for position columns (Horiba LabSpec map export:
    ``<tab><tab>w1 w2 ...`` then ``x y i1 i2 ...``).
``long`` / tidy (one point per row)
    ``[spectrum,] [name,] [x, y, ...,] wavenumber, intensity`` -- also the
    headerless Renishaw WiRE exports ``wl int``, ``x wl int``, ``y x wl int``.
    Several value columns (e.g. ``baseline, cosmic_rays, noise, peaks``)
    give several arrays.

Text readers additionally skip instrument preambles and trailers (Ocean
Insight ``>>>>>Begin Spectral Data<<<<<``, RRUFF ``##`` headers, LabSpec
``#key=value`` lines), detect the delimiter and accept decimal commas.
"""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._record import INTENSITY, Record, default_names, is_monotonic

COMPONENT_NAMES = ("spectra", "baseline", "cosmic_rays", "noise", "peaks")

_WN_ALIASES = {"wavenumber", "wavenumbers", "ramanshift", "raman_shift", "shift", "wave",
               "#wave", "wl", "wavelength", "x-axis", "xaxis", "frequency", "cm-1", "cm^-1",
               "1/cm", "energy", "pixel", "channel", "index"}
_INT_ALIASES = {"intensity", "#intensity", "int", "counts", "count", "signal", "y", "y-axis",
                "yaxis", "absorbance", "spectrum", "value", "cps"}
_ID_ALIASES = {"spectrum", "spectrum_id", "spectrumid", "id", "spec", "sample", "spectrum_index"}
_NAME_ALIASES = {"name", "spectrum_name", "label", "title"}
_POS_ALIASES = {"x", "y", "z", "#x", "#y", "#z", "t", "time", "#time", "position", "pos_x",
                "pos_y", "pos_z", "xpos", "ypos", "zpos", "depth"}

_NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_DEC_COMMA = re.compile(r"^[+-]?\d+,\d+(?:[eE][+-]?\d+)?$")


def _key(name) -> str:
    return re.sub(r"\s+", "", str(name)).strip().lower()


def _to_float(s) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float, np.number)):
        return float(s)
    s = str(s).strip().strip('"')
    if s == "":
        return None
    if _NUMBER.match(s):
        return float(s)
    if s.lower() in ("nan", "na", "?"):
        return float("nan")
    return None


# ---------------------------------------------------------------------------
# Generic cell grid -> columns
# ---------------------------------------------------------------------------
def _is_data_row(cells: Sequence) -> bool:
    vals = [c for c in cells if not (c is None or str(c).strip() == "")]
    if not vals:
        return False
    numeric = sum(_to_float(c) is not None for c in vals)
    return numeric >= max(1, len(vals) // 2 + (len(vals) % 2))


def grid_to_columns(rows: List[List], has_header: Optional[bool] = None):
    """Find the data block in a grid of cells and return ``(header, columns)``.

    ``columns`` is an ordered dict ``{name: ndarray}`` (float, or object for
    text columns); ``header`` is ``None`` when the block has no header row.
    """
    rows = [r for r in rows if r is not None]
    flags = [_is_data_row(r) for r in rows]
    # longest run of consecutive data rows
    best, start = (0, 0), None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best[1] == best[0]:
        raise ValueError("no numeric data found")
    data = rows[best[0]:best[1]]
    width = max(len(r) for r in data)
    header = None
    if has_header is not False and best[0] > 0:
        cand = [str(c).strip() if c is not None else "" for c in rows[best[0] - 1]]
        if 1 < len(cand) and len([c for c in cand if c]) >= max(1, width - 1) and not _is_data_row(cand):
            header = cand + [""] * (width - len(cand))
    if has_header is True and header is None:
        header, data = [str(c) for c in data[0]], data[1:]
    names = header if header is not None else [str(i) for i in range(width)]
    names = [n if n else f"column_{i}" for i, n in enumerate(names)]
    cols: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for j, name in enumerate(names[:width]):
        raw = [r[j] if j < len(r) else None for r in data]
        vals = [_to_float(c) for c in raw]
        if all(v is not None or (c is None or str(c).strip() == "") for v, c in zip(vals, raw)):
            cols[name] = np.array([np.nan if v is None else v for v in vals], dtype=np.float64)
        else:
            cols[name] = np.array([("" if c is None else str(c).strip()) for c in raw], dtype=object)
    # drop trailing all-empty columns (e.g. a trailing delimiter)
    for name in list(cols)[::-1]:
        c = cols[name]
        if c.dtype != object and np.all(np.isnan(c)) and name.startswith("column_"):
            del cols[name]
        else:
            break
    return header, cols


# ---------------------------------------------------------------------------
# Columns -> Record
# ---------------------------------------------------------------------------
def _find(cols, aliases) -> Optional[str]:
    for name in cols:
        if _key(name) in aliases:
            return name
    return None


def _numeric(cols) -> List[str]:
    return [n for n, c in cols.items() if c.dtype != object]


def _periodic_axis(x: np.ndarray) -> Optional[int]:
    """If ``x`` is a sequence that repeats (block length L >= 3), return L."""
    n = len(x)
    if n < 6:
        return None
    first = x[0]
    restarts = np.flatnonzero(np.isclose(x, first))
    if len(restarts) < 2:
        return None
    L = int(restarts[1])
    if L < 3 or n % L != 0:
        return None
    blocks = x.reshape(-1, L)
    if np.allclose(blocks, blocks[0]) and is_monotonic(blocks[0]):
        return L
    return None


def _long_to_record(cols, wn_name: str, value_names: List[str], id_name, name_name,
                    pos_names: List[str]) -> Record:
    wn = cols[wn_name]
    n = len(wn)
    # group rows into spectra
    if id_name is not None:
        keys = cols[id_name]
    elif pos_names:
        keys = np.array([tuple(cols[p][i] for p in pos_names) for i in range(n)], dtype=object)
    else:
        L = _periodic_axis(wn)
        if L is None:  # restart whenever the axis changes direction
            d = np.sign(np.diff(wn))
            main = np.sign(np.median(d)) or 1
            breaks = np.flatnonzero(d != main) + 1
            keys = np.zeros(n, dtype=int)
            for b in breaks:
                keys[b:] += 1
        else:
            keys = np.repeat(np.arange(n // L), L)
    order, groups = [], {}
    for i, k in enumerate(keys.tolist() if hasattr(keys, "tolist") else keys):
        k = tuple(k) if isinstance(k, (list, np.ndarray)) else k
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(i)
    idx = [np.asarray(groups[k]) for k in order]
    xs = [wn[i] for i in idx]
    same = all(len(x) == len(xs[0]) and np.allclose(x, xs[0]) for x in xs)
    arrays = {}
    for v in value_names:
        ys = [cols[v][i] for i in idx]
        arrays[v] = np.vstack(ys) if same else ys
    names = None
    if name_name is not None:
        names = [str(cols[name_name][i[0]]) for i in idx]
    elif id_name is not None and cols[id_name].dtype == object:
        names = [str(k) for k in order]
    positions = None
    if pos_names:
        positions = np.array([[cols[p][i[0]] for p in pos_names] for i in idx], dtype=np.float64)
    return Record(arrays, xs[0] if same else xs, names, positions,
                  [_key(p).lstrip("#") for p in pos_names], None, "", "", {"layout": "long"})


def table_to_record(cols: "OrderedDict[str, np.ndarray]", header, layout: str = "auto") -> Record:
    """Interpret a table as spectra (see the module docstring for layouts)."""
    if not cols:
        raise ValueError("empty table")
    numeric = _numeric(cols)
    if not numeric:
        raise ValueError("table has no numeric columns")

    # ---- long (tidy) layout ----------------------------------------------
    if layout in ("auto", "long"):
        wn_name = _find(cols, _WN_ALIASES) if header is not None else None
        if wn_name is not None and wn_name in numeric:
            value_names = [n for n in numeric if n != wn_name and (
                _key(n) in COMPONENT_NAMES or
                (_key(n) in _INT_ALIASES and _key(n) not in _ID_ALIASES | _POS_ALIASES))]
            if not value_names:  # e.g. "x, y" columns: y is the intensity
                value_names = [n for n in numeric if n != wn_name and _key(n) in _INT_ALIASES
                               and _key(n) not in _ID_ALIASES]
            id_name = next((n for n in cols if _key(n) in _ID_ALIASES and n not in value_names), None)
            name_name = next((n for n in cols if _key(n) in _NAME_ALIASES), None)
            pos_names = [n for n in numeric if _key(n) in _POS_ALIASES and n != wn_name
                         and n not in value_names]
            repeats = len(np.unique(cols[wn_name])) < len(cols[wn_name])
            if value_names and (repeats or id_name or layout == "long"):
                rec = _long_to_record(cols, wn_name, value_names, id_name, name_name, pos_names)
                if list(rec.arrays) == [value_names[0]] and _key(value_names[0]) not in COMPONENT_NAMES:
                    rec.arrays = {INTENSITY: rec.arrays[value_names[0]]}
                else:
                    rec.arrays = {_key(k): v for k, v in rec.arrays.items()}
                return rec
        # headerless Renishaw-style exports: [pos...] wl int with a repeating wl column
        if header is None and len(numeric) >= 3 and len(numeric) == len(cols):
            names = list(cols)
            wl, it = names[-2], names[-1]
            L = _periodic_axis(cols[wl])
            if L is not None or layout == "long":
                pos = names[:-2]
                rec = _long_to_record(cols, wl, [it], None, None, pos)
                rec.arrays = {INTENSITY: rec.arrays[it]}
                rec.position_names = (["x", "y", "z"][: len(pos)] if len(pos) != 2 else ["y", "x"])
                return rec

    numcols = [cols[n] for n in numeric]
    table = np.column_stack(numcols)
    nrow, ncol = table.shape
    names_num = numeric

    def as_columns():
        first = table[:, 0]
        if ncol >= 2 and is_monotonic(first):
            spectra = table[:, 1:].T
            nm = names_num[1:] if header is not None else None
            return Record({INTENSITY: spectra}, first, nm, meta={"layout": "wide"})
        return Record({INTENSITY: table.T}, None, names_num if header is not None else None,
                      meta={"layout": "wide"})

    def as_rows():
        first = table[0]
        lead = 0
        while lead < ncol and np.isnan(first[lead]):
            lead += 1
        if nrow >= 2 and is_monotonic(first[lead:]):
            wn = first[lead:]
            body = table[1:]
            positions = body[:, :lead] if lead else None
            pos_names = ["x", "y", "z"][:lead] if lead != 2 else ["x", "y"]
            return Record({INTENSITY: body[:, lead:]}, wn, None, positions,
                          pos_names if lead else [], meta={"layout": "rows"})
        if header is not None and all(_to_float(h) is not None for h in names_num):
            return Record({INTENSITY: table}, np.array([float(h) for h in names_num]), None,
                          meta={"layout": "rows"})
        return Record({INTENSITY: table}, None, None, meta={"layout": "rows"})

    if layout == "columns" or layout == "wide":
        return as_columns()
    if layout == "rows":
        return as_rows()
    # auto
    first_row = table[0]
    lead = int(np.argmax(~np.isnan(first_row))) if np.any(np.isnan(first_row[:1])) else 0
    if nrow >= 2 and lead > 0 and is_monotonic(first_row[lead:]):
        return as_rows()  # LabSpec map export
    if header is not None and ncol > 2 and all(_to_float(h) is not None for h in names_num) \
            and is_monotonic([float(h) for h in names_num]):
        return as_rows()  # numeric header row = wavenumbers
    if ncol <= 2 or nrow > ncol:
        return as_columns()
    return as_rows()


# ---------------------------------------------------------------------------
# Record -> columns (writing)
# ---------------------------------------------------------------------------
def record_to_long(record: Record) -> "OrderedDict[str, np.ndarray]":
    """Tidy table: spectrum, [name], [positions], wavenumber, <one column per array>."""
    arrays = record.arrays
    first = next(iter(arrays.values()))
    n = len(first)
    lengths = [len(np.ravel(first[i])) for i in range(n)]
    if isinstance(record.wavenumbers, list):
        wns = [np.asarray(w) for w in record.wavenumbers]
    elif record.wavenumbers is not None:
        wns = [np.asarray(record.wavenumbers)] * n
    else:
        wns = [np.arange(L, dtype=np.float64) for L in lengths]
    cols: "OrderedDict[str, np.ndarray]" = OrderedDict()
    cols["spectrum"] = np.concatenate([np.full(L, i + 1) for i, L in enumerate(lengths)])
    if record.names:
        cols["name"] = np.concatenate([np.full(L, record.names[i], dtype=object)
                                       for i, L in enumerate(lengths)])
    if record.positions is not None:
        pn = record.position_names or [f"pos_{j}" for j in range(record.positions.shape[1])]
        for j, p in enumerate(pn):
            cols[p] = np.concatenate([np.full(L, record.positions[i, j]) for i, L in enumerate(lengths)])
    cols["wavenumber"] = np.concatenate(wns)
    for k, a in arrays.items():
        cols[k] = np.concatenate([np.ravel(a[i]) for i in range(n)])
    return cols


def record_to_wide(record: Record, array: Optional[str] = None) -> "OrderedDict[str, np.ndarray]":
    """One spectrum per column, wavenumber column first (single array, equal lengths)."""
    key = array or next(iter(record.arrays))
    a = record.arrays[key]
    if isinstance(a, list):
        raise ValueError("spectra of different lengths cannot be written in wide layout; use layout='long'")
    a = np.atleast_2d(a)
    names = record.names or default_names(len(a))
    cols: "OrderedDict[str, np.ndarray]" = OrderedDict()
    if record.wavenumbers is not None:
        cols["wavenumber"] = np.asarray(record.wavenumbers)
    else:
        cols["index"] = np.arange(a.shape[1], dtype=np.float64)
    used = set(cols)
    for nm, row in zip(names, a):
        nm = str(nm)
        while nm in used:
            nm += "_"
        used.add(nm)
        cols[nm] = row
    return cols


# ---------------------------------------------------------------------------
# Delimited text
# ---------------------------------------------------------------------------
_TEXT_DELIMS = ("\t", ";", ",", "|")


def _sniff(lines: List[str]) -> Optional[str]:
    sample = [ln for ln in lines if ln.strip()][:400]
    best, best_score = None, 0
    for d in _TEXT_DELIMS:
        counts = [ln.count(d) for ln in sample]
        if not counts:
            continue
        mode = max(set(counts), key=counts.count)
        score = counts.count(mode) * (mode > 0)
        if mode > 0 and score > best_score:
            best, best_score = d, score
    if best == "," and sample:
        # "1,5 2,5" with whitespace separation uses decimal commas
        if all(re.fullmatch(r"\s*[+-]?\d+,\d+(\s+[+-]?\d+,\d+)*\s*", ln) for ln in sample[-20:]):
            return None
    return best


def read_text_table(path, delimiter=None, decimal=None, comments="#", skip_header=0,
                    encoding=None):
    """Read a delimited text file into ``(header, columns)``."""
    raw = Path(path).read_bytes()
    text = None
    for enc in ([encoding] if encoding else []) + ["utf-8-sig", "utf-16", "latin-1"]:
        try:
            text = raw.decode(enc)
            if "\x00" in text and enc != "utf-16":
                continue
            break
        except (UnicodeDecodeError, LookupError):
            continue
    lines = text.splitlines()[skip_header:]
    # Ocean Insight / OceanView: data between Begin/End Spectral Data markers
    begin = next((i for i, l in enumerate(lines) if "begin spectral data" in l.lower()), None)
    if begin is not None:
        end = next((i for i, l in enumerate(lines) if "end spectral data" in l.lower()), len(lines))
        lines = lines[begin + 1:end]
    if comments:
        kept = []
        for l in lines:
            s = l.lstrip()
            if s.startswith(comments) and not re.match(r"^#\s*(wave|x|y|z|intensity|time)\b", s, re.I):
                continue
            kept.append(l)
        lines = kept
    delim = delimiter if delimiter is not None else _sniff(lines)
    rows = []
    for l in lines:
        if not l.strip():
            continue
        if delim is None:
            cells = l.split()
        else:
            cells = next(csv.reader([l], delimiter=delim)) if '"' in l else l.split(delim)
        cells = [c.strip() for c in cells]
        rows.append(cells)
    use_comma = decimal == "," or (decimal is None and delim != "," and any(
        _DEC_COMMA.match(c) for r in rows[-50:] for c in r))
    if use_comma:
        rows = [[c.replace(".", "").replace(",", ".") if _DEC_COMMA.match(c) else c for c in r]
                for r in rows]
    return grid_to_columns(rows)


def write_text_table(path, cols: "OrderedDict[str, np.ndarray]", delimiter=",", decimal="."):
    path = Path(path)
    names = list(cols)
    n = len(next(iter(cols.values())))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=delimiter)
        w.writerow(names)
        arrs = [cols[k] for k in names]
        for i in range(n):
            row = []
            for a in arrs:
                v = a[i]
                if isinstance(v, (float, np.floating)):
                    s = repr(float(v))
                    if decimal == ",":
                        s = s.replace(".", ",")
                    row.append(s)
                elif isinstance(v, (int, np.integer)):
                    row.append(str(int(v)))
                else:
                    row.append(str(v))
            w.writerow(row)
    return path


# ---------------------------------------------------------------------------
# pandas-based formats (Excel, Parquet, Feather)
# ---------------------------------------------------------------------------
def _missing(package, error):
    """Message for an optional package that is missing or fails to import."""
    if importlib.util.find_spec(package) is None:
        return f"This format needs {package}: pip install 'radar-raman[formats]'"
    return (f"{package} is installed but cannot be imported ({error}). "
            f"Reinstall it in this Python: {sys.executable} -m pip install --force-reinstall {package}")


def _pyarrow():
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise ImportError(_missing("pyarrow", e)) from e


def _pandas():
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError(_missing("pandas", e)) from e
    return pd


def _df_to_columns(df):
    cols: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for c in df.columns:
        s = df[c]
        if s.dtype.kind in "biuf":
            cols[str(c)] = s.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            cols[str(c)] = s.astype(str).to_numpy(dtype=object)
    return cols


def read_excel_table(path, sheet=0):
    pd = _pandas()
    df = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)
    rows = df.where(df.notna(), None).values.tolist()
    return grid_to_columns(rows)


def excel_sheet_names(path) -> List[str]:
    pd = _pandas()
    return list(pd.ExcelFile(path).sheet_names)


def write_excel(path, sheets: Dict[str, "OrderedDict[str, np.ndarray]"], meta: dict):
    pd = _pandas()
    with pd.ExcelWriter(path) as xw:
        for name, cols in sheets.items():
            pd.DataFrame(cols).to_excel(xw, sheet_name=name[:31], index=False)
        if meta:
            pd.DataFrame({"key": list(meta), "value": [str(v) for v in meta.values()]}).to_excel(
                xw, sheet_name="radar_metadata", index=False)
    return Path(path)


def read_parquet_table(path):
    pd = _pandas()
    _pyarrow()
    df = pd.read_parquet(path)
    return list(map(str, df.columns)), _df_to_columns(df), _parquet_meta(path)


def _parquet_meta(path) -> dict:
    try:
        import json

        import pyarrow.parquet as pq

        md = pq.read_schema(path).metadata or {}
        raw = md.get(b"radar")
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def write_parquet(path, cols, meta: dict):
    pd = _pandas()
    import json

    df = pd.DataFrame(cols)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df, preserve_index=False)
        md = dict(table.schema.metadata or {})
        md[b"radar"] = json.dumps(meta, default=str).encode()
        pq.write_table(table.replace_schema_metadata(md), path)
    except ImportError as e:  # pragma: no cover
        raise ImportError(_missing("pyarrow", e)) from e
    return Path(path)


def read_feather_table(path):
    pd = _pandas()
    _pyarrow()
    df = pd.read_feather(path)
    return list(map(str, df.columns)), _df_to_columns(df)


def write_feather(path, cols):
    pd = _pandas()
    _pyarrow()
    pd.DataFrame(cols).to_feather(path)
    return Path(path)


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------
def read_sqlite_table(path, table: Optional[str] = None, query: Optional[str] = None):
    import json
    import sqlite3

    # immutable=1: no locking, so files on network / synced folders can be read
    con = sqlite3.connect(f"file:{Path(path).resolve().as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name")]
        meta = {}
        if "radar_metadata" in tables:
            for k, v in con.execute("SELECT key, value FROM radar_metadata"):
                try:
                    meta[k] = json.loads(v)
                except (TypeError, ValueError):
                    meta[k] = v
        if query is None:
            if table is None:
                cands = [t for t in tables if t != "radar_metadata" and not t.startswith("sqlite_")]
                if not cands:
                    raise ValueError(f"{path}: database has no tables")
                table = "spectra" if "spectra" in cands else cands[0]
            if table not in tables:
                raise ValueError(f"{path}: no table {table!r}; tables: {tables}")
            query = f'SELECT * FROM "{table}"'
        cur = con.execute(query)
        names = [d[0] for d in cur.description]
        rows = cur.fetchall()
    finally:
        con.close()
    cols: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for j, nm in enumerate(names):
        vals = [r[j] for r in rows]
        if all(v is None or isinstance(v, (int, float)) for v in vals):
            cols[nm] = np.array([np.nan if v is None else float(v) for v in vals], dtype=np.float64)
        else:
            cols[nm] = np.array(["" if v is None else str(v) for v in vals], dtype=object)
    return names, cols, meta, tables


def write_sqlite(path, cols, meta: dict, table: str = "spectra", if_exists: str = "replace"):
    """Write a table (and metadata) to a SQLite file.

    The database is built in a local temporary file and then moved to
    ``path``: SQLite needs file locking, which network drives, mounted and
    cloud-synced folders often do not provide.
    """
    import shutil
    import tempfile

    path = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_db = Path(tmp) / "radar_raman.db"
        if if_exists == "append" and path.exists():
            shutil.copyfile(path, tmp_db)
        elif if_exists == "fail" and path.exists():
            raise FileExistsError(path)
        _write_sqlite_local(tmp_db, cols, meta, table, if_exists)
        shutil.copyfile(tmp_db, path)
    return path


def _write_sqlite_local(path, cols, meta: dict, table: str, if_exists: str):
    import json
    import sqlite3

    con = sqlite3.connect(path)
    try:
        if if_exists == "replace":
            con.execute(f'DROP TABLE IF EXISTS "{table}"')
        defs = []
        for k, v in cols.items():
            typ = "TEXT" if v.dtype == object else (
                "INTEGER" if np.issubdtype(v.dtype, np.integer) else "REAL")
            defs.append(f'"{k}" {typ}')
        con.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({", ".join(defs)})')
        names = list(cols)
        n = len(cols[names[0]])

        def conv(v):
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating, float)):
                return None if np.isnan(v) else float(v)
            return v

        con.executemany(
            f'INSERT INTO "{table}" ({", ".join(chr(34) + k + chr(34) for k in names)}) '
            f'VALUES ({", ".join("?" * len(names))})',
            ([conv(cols[k][i]) for k in names] for i in range(n)),
        )
        if "spectrum" in cols:
            con.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_spectrum" ON "{table}"(spectrum)')
        con.execute("CREATE TABLE IF NOT EXISTS radar_metadata (key TEXT PRIMARY KEY, value TEXT)")
        con.executemany("INSERT OR REPLACE INTO radar_metadata VALUES (?, ?)",
                        [(k, json.dumps(v, default=str)) for k, v in meta.items()])
        con.commit()
    finally:
        con.close()
