# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""JCAMP-DX spectra (``.jdx``, ``.dx``, ``.jcamp``): native reader and writer.

Reader
------
* versions 4.24 and 5.x, single blocks and compound files (``##BLOCKS``,
  ``LINK`` blocks);
* ``##XYDATA=(X++(Y..Y))`` in AFFN (plain numbers) and in all ASDF
  compressed forms (SQZ, DIF with Y-checks, DUP);
* ``##XYPOINTS=(XY..XY)`` and ``##PEAK TABLE=(XY..XY)``;
* ``XFACTOR``/``YFACTOR``, ``FIRSTX``/``LASTX``/``NPOINTS`` scaling.

``##NTUPLES`` (multi-dimensional NMR/MS pages) is not supported.

Writer
------
``##XYDATA=(X++(Y..Y))`` in AFFN for evenly spaced axes and
``##XYPOINTS=(XY..XY)`` otherwise; several spectra become a compound file
with one block per spectrum.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ._record import INTENSITY, Record

_LABEL = re.compile(r"^\s*##\s*([^=]*)=(.*)$")
_NUM = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
# a line of plain (AFFN) numbers separated by blanks, commas or semicolons
_AFFN_LINE = re.compile(rf"^\s*(?:(?:{_NUM}|\?)(?:[\s,;]+|(?=[+-])|$))*\s*$")
_TOKEN = re.compile(rf"{_NUM}|\?")
# in compressed (ASDF) lines E/e are SQZ digits, so numbers carry no exponent
_ASDF_TOKEN = re.compile(r"[@A-Ia-i%J-Rj-rS-Zs][0-9]*(?:\.[0-9]*)?|[+-]?(?:\d+\.?\d*|\.\d+)|\?")

_SQZ = {c: (i + 1) for i, c in enumerate("ABCDEFGHI")}
_SQZ.update({c: -(i + 1) for i, c in enumerate("abcdefghi")})
_SQZ["@"] = 0
_DIF = {c: (i + 1) for i, c in enumerate("JKLMNOPQR")}
_DIF.update({c: -(i + 1) for i, c in enumerate("jklmnopqr")})
_DIF["%"] = 0
_DUP = {c: (i + 1) for i, c in enumerate("STUVWXYZ")}
_DUP["s"] = 9


def _norm_label(label: str) -> str:
    return re.sub(r"[\s\-/_]", "", label).upper()


def _num(s, default=None):
    try:
        return float(str(s).strip().split()[0])
    except (ValueError, IndexError):
        return default


# ---------------------------------------------------------------------------
# ASDF decoding
# ---------------------------------------------------------------------------
def _decode_line(line: str) -> Tuple[List[float], bool]:
    """Decode one XYDATA line -> (values including the leading X, ends_in_dif)."""
    line = line.split("$$", 1)[0]  # comments
    if _AFFN_LINE.match(line):
        vals = [float(t) if t != "?" else np.nan for t in _TOKEN.findall(line)]
        return vals, False
    values: List[float] = []
    last_kind = None  # "abs" or "dif"
    last_dif = 0.0
    for tok in _ASDF_TOKEN.findall(line):
        c = tok[0]
        if c in _SQZ:
            values.append(float(f"{_SQZ[c]}{tok[1:]}" if _SQZ[c] >= 0 else f"-{abs(_SQZ[c])}{tok[1:]}"))
            last_kind = "abs"
        elif c in _DIF:
            d = _DIF[c]
            dif = float(f"{d}{tok[1:]}" if d >= 0 else f"-{abs(d)}{tok[1:]}")
            values.append(values[-1] + dif)
            last_kind, last_dif = "dif", dif
        elif c in _DUP:
            count = int(f"{_DUP[c]}{tok[1:]}") - 1
            for _ in range(count):
                values.append(values[-1] + last_dif if last_kind == "dif" else values[-1])
        elif tok == "?":
            values.append(np.nan)
            last_kind = "abs"
        else:  # AFFN number embedded in an ASDF line
            values.append(float(tok))
            last_kind = "abs"
    return values, last_kind == "dif"


def _decode_xydata(lines: List[str]) -> List[float]:
    ys: List[float] = []
    prev_dif = False
    for line in lines:
        if not line.strip():
            continue
        vals, ends_dif = _decode_line(line)
        if len(vals) < 2:
            continue
        y = vals[1:]
        if prev_dif and ys:  # Y-check value duplicates the last Y of the previous line
            y = y[1:]
        ys.extend(y)
        prev_dif = ends_dif
    return ys


def _decode_xy_pairs(lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    nums = [float(t) for line in lines for t in _TOKEN.findall(line.split("$$", 1)[0]) if t != "?"]
    arr = np.asarray(nums[: len(nums) // 2 * 2], dtype=np.float64).reshape(-1, 2)
    return arr[:, 0], arr[:, 1]


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
def _parse_blocks(text: str) -> List[Dict]:
    """Split a JCAMP file into data blocks: {labels: {...}, data_label, data_lines}."""
    blocks: List[Dict] = []
    stack: List[Dict] = []
    current = None
    for raw in text.splitlines():
        m = _LABEL.match(raw)
        if m:
            label, value = _norm_label(m.group(1)), m.group(2).strip()
            if label == "TITLE":
                current = {"labels": {"TITLE": value}, "data_label": None, "data_lines": [],
                           "parent": stack[-1]["labels"] if stack else {}}
                stack.append(current)
                continue
            if current is None:
                current = {"labels": {}, "data_label": None, "data_lines": [], "parent": {}}
                stack.append(current)
            if label == "END":
                blk = stack.pop() if stack else current
                if blk.get("data_label"):
                    blocks.append(blk)
                current = stack[-1] if stack else None
                continue
            current["labels"][label] = value
            if label in ("XYDATA", "XYPOINTS", "PEAKTABLE", "DATATABLE"):
                current["data_label"] = (label, value)
                current["data_lines"] = []
            continue
        if current is not None and current.get("data_label"):
            current["data_lines"].append(raw)
    if current is not None and current.get("data_label") and current not in blocks:
        blocks.append(current)
    return blocks


def _block_to_xy(blk: Dict) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    lab = {**blk["parent"], **blk["labels"]}
    kind, form = blk["data_label"]
    xf = _num(lab.get("XFACTOR"), 1.0)
    yf = _num(lab.get("YFACTOR"), 1.0)
    form_compact = form.replace(" ", "").upper()
    pairs = "XY..XY" in form_compact or "XYXY" in form_compact
    if (kind == "XYDATA" and not pairs) or (kind == "DATATABLE" and "X++" in form_compact):
        y = np.asarray(_decode_xydata(blk["data_lines"]), dtype=np.float64) * yf
        first = _num(lab.get("FIRSTX"))
        last = _num(lab.get("LASTX"))
        npts = int(_num(lab.get("NPOINTS"), len(y)))
        if npts and len(y) > npts:
            y = y[:npts]
        if first is None or last is None:
            vals0 = [_decode_line(l)[0] for l in blk["data_lines"] if l.strip()]
            first = vals0[0][0] * xf
            last = first + (len(y) - 1) * (_num(lab.get("DELTAX"), xf))
        x = np.linspace(first, last, len(y))
    elif kind in ("XYPOINTS", "PEAKTABLE", "DATATABLE") or pairs:
        # (XY..XY) pairs -- also found as the non-standard "##XYDATA=(XY..XY)"
        x, y = _decode_xy_pairs(blk["data_lines"])
        x, y = x * xf, y * yf
    else:  # pragma: no cover
        raise ValueError(f"unsupported JCAMP data record {kind}")
    return x, y, lab


def read_jcamp(path) -> Record:
    """Read a JCAMP-DX file into a :class:`Record` (one spectrum per data block)."""
    text = Path(path).read_text(encoding="latin-1", errors="replace")
    if "##NTUPLES" in text.upper().replace(" ", ""):
        raise ValueError(f"{path}: JCAMP-DX NTUPLES files are not supported")
    blocks = _parse_blocks(text)
    if not blocks:
        raise ValueError(f"{path}: no XYDATA/XYPOINTS/PEAK TABLE data found")
    xs, ys, names, labels = [], [], [], []
    for blk in blocks:
        x, y, lab = _block_to_xy(blk)
        xs.append(x)
        ys.append(y)
        names.append(lab.get("TITLE", f"spectrum_{len(names) + 1}"))
        labels.append(lab)
    same = all(len(x) == len(xs[0]) and np.allclose(x, xs[0]) for x in xs)
    arrays = {INTENSITY: np.vstack(ys) if same else ys}
    lab0 = labels[0]
    meta = {"format": "JCAMP-DX", "jcamp_version": lab0.get("JCAMPDX", ""),
            "data_type": lab0.get("DATATYPE", ""),
            "labels": {k: v for k, v in lab0.items() if len(v) < 200}}
    return Record(arrays, xs[0] if same else xs, names, None, [], None,
                  lab0.get("XUNITS", ""), lab0.get("YUNITS", ""), meta)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def _fmt(v: float) -> str:
    return "?" if not np.isfinite(v) else repr(float(v))


def _block(title: str, x: np.ndarray, y: np.ndarray, x_units: str, y_units: str,
           data_type: str, per_line: int = 8) -> List[str]:
    out = [f"##TITLE={title}", "##JCAMP-DX=5.01", f"##DATA TYPE={data_type}",
           "##ORIGIN=RADAR", "##OWNER=", f"##XUNITS={x_units or '1/CM'}",
           f"##YUNITS={y_units or 'ARBITRARY UNITS'}", "##XFACTOR=1", "##YFACTOR=1",
           f"##FIRSTX={_fmt(x[0])}", f"##LASTX={_fmt(x[-1])}", f"##NPOINTS={len(y)}",
           f"##FIRSTY={_fmt(y[0])}", f"##MINY={_fmt(np.nanmin(y))}", f"##MAXY={_fmt(np.nanmax(y))}"]
    n = len(x)
    even = n > 1 and np.allclose(np.diff(x), (x[-1] - x[0]) / (n - 1), rtol=1e-9, atol=1e-12)
    if even:
        out.append(f"##DELTAX={_fmt((x[-1] - x[0]) / (n - 1))}")
        out.append("##XYDATA=(X++(Y..Y))")
        for i in range(0, n, per_line):
            out.append(" ".join([_fmt(x[i])] + [_fmt(v) for v in y[i:i + per_line]]))
    else:
        out.append("##XYPOINTS=(XY..XY)")
        out.extend(f"{_fmt(a)}, {_fmt(b)}" for a, b in zip(x, y))
    out.append("##END=")
    return out


def write_jcamp(path, record: Record, data_type: str = "RAMAN SPECTRUM") -> Path:
    """Write the single array of ``record`` as a JCAMP-DX 5.01 file."""
    if len(record.arrays) != 1:
        raise ValueError("write_jcamp expects a record with one array")
    ys = [np.asarray(y, dtype=np.float64).ravel() for y in next(iter(record.arrays.values()))]
    if isinstance(record.wavenumbers, list):
        xs = [np.asarray(x, dtype=np.float64) for x in record.wavenumbers]
    elif record.wavenumbers is not None:
        xs = [np.asarray(record.wavenumbers, dtype=np.float64)] * len(ys)
    else:
        xs = [np.arange(len(y), dtype=np.float64) for y in ys]
    names = record.names or [f"spectrum_{i + 1}" for i in range(len(ys))]
    x_units = _jcamp_units(record.x_units)
    lines: List[str] = []
    if len(ys) == 1:
        lines = _block(names[0], xs[0], ys[0], x_units, record.y_units, data_type)
    else:
        lines = [f"##TITLE={Path(path).stem}", "##JCAMP-DX=5.01", "##DATA TYPE=LINK",
                 f"##BLOCKS={len(ys)}"]
        for i, (x, y) in enumerate(zip(xs, ys)):
            blk = _block(names[i], x, y, x_units, record.y_units, data_type)
            blk.insert(1, f"##BLOCK_ID={i + 1}")
            lines.extend(blk)
        lines.append("##END=")
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="latin-1", errors="replace")
    return path


def _jcamp_units(u: str) -> str:
    u = (u or "").lower()
    if "cm" in u and ("-1" in u or "^-1" in u or "1/" in u or "shift" in u or "wavenumber" in u):
        return "1/CM"
    if u in ("nm", "nanometers"):
        return "NANOMETERS"
    return u.upper() if u else "1/CM"
