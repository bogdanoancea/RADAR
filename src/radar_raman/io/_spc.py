# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Thermo Galactic GRAMS ``.spc`` files (native reader and writer).

Implemented from the public SPC file-format specification (Galactic
Industries, 1996/1997):

* new format, little-endian (``fversn = 0x4B``): single files, multifiles
  (``TMULTI``), evenly spaced X, a common X array (``TXVALS``) or one X array
  per sub-file (``TXYXYS``), 16-bit / 32-bit integer or IEEE float Y;
* old format (``fversn = 0x4D``), read only.

The new-format big-endian variant (``0x4C``) is rare and not supported.
The writer produces new-format files with float Y and one sub-file per
spectrum.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import List

import numpy as np

from ._record import INTENSITY, Record

# flag bits of ftflgs
TSPREC, TCGRAM, TMULTI, TRANDM, TORDRD, TALABS, TXYXYS, TXVALS = (1, 2, 4, 8, 16, 32, 64, 128)

X_UNITS = {
    0: "arbitrary", 1: "cm-1", 2: "um", 3: "nm", 4: "s", 5: "min", 6: "Hz", 7: "kHz",
    8: "MHz", 9: "m/z", 10: "ppm", 11: "days", 12: "years", 13: "Raman shift (cm-1)",
    14: "eV", 15: "text labels", 16: "diode number", 17: "channel", 18: "degrees",
    19: "deg F", 20: "deg C", 21: "K", 22: "data points", 23: "ms", 24: "us", 25: "ns",
    26: "GHz", 27: "cm", 28: "m", 29: "mm", 30: "h", 255: "double interferogram",
}
Y_UNITS = {
    0: "arbitrary intensity", 1: "interferogram", 2: "absorbance", 3: "Kubelka-Munk",
    4: "counts", 5: "V", 6: "degrees", 7: "mA", 8: "mm", 9: "mV", 10: "log(1/R)",
    11: "percent", 12: "intensity", 13: "relative intensity", 14: "energy", 16: "dB",
    19: "deg F", 20: "deg C", 21: "K", 22: "index of refraction", 23: "extinction coeff.",
    24: "real", 25: "imaginary", 26: "complex", 128: "transmission", 129: "reflectance",
    130: "valley", 131: "emission",
}

_NEW_HDR = struct.Struct("<BBBbiddiBBBBi9s9sH32s130s30siiBBHf48sfifB187s")
_OLD_HDR = struct.Struct("<BBhfffBBHBBBB8shh28s130s30s")
_SUB_HDR = struct.Struct("<BbHfffIIf4s")
assert _NEW_HDR.size == 512 and _OLD_HDR.size == 224 and _SUB_HDR.size == 32


def _text(b: bytes) -> str:
    return b.split(b"\x00", 1)[0].decode("latin-1", errors="replace").strip()


def _y_values(buf: bytes, pos: int, npts: int, exp: int, flags: int):
    if exp == -128:  # IEEE float
        y = np.frombuffer(buf, "<f4", npts, pos).astype(np.float64)
        return y, pos + 4 * npts
    if flags & TSPREC:
        y = np.frombuffer(buf, "<i2", npts, pos).astype(np.float64) * 2.0 ** (exp - 16)
        return y, pos + 2 * npts
    y = np.frombuffer(buf, "<i4", npts, pos).astype(np.float64) * 2.0 ** (exp - 32)
    return y, pos + 4 * npts


def read_spc(path) -> Record:
    """Read a ``.spc`` file into a :class:`Record`."""
    buf = Path(path).read_bytes()
    if len(buf) < 2:
        raise ValueError(f"{path}: not an SPC file")
    version = buf[1]
    if version == 0x4B:
        return _read_new(buf, path)
    if version == 0x4D:
        return _read_old(buf, path)
    if version == 0x4C:
        raise ValueError(f"{path}: big-endian SPC files (0x4C) are not supported")
    raise ValueError(f"{path}: unknown SPC version byte 0x{version:02x}")


def _read_new(buf: bytes, path) -> Record:
    (flags, _ver, _exper, fexp, fnpts, ffirst, flast, fnsub, fxtype, fytype, fztype, _fpost,
     _fdate, _fres, fsource, _fpeakpt, _spare, fcmnt, fcatxt, flogoff, _fmods, _fprocs,
     _flevel, _fsampin, _ffactor, _fmethod, _fzinc, _fwplanes, _fwinc, _fwtype,
     _reserved) = _NEW_HDR.unpack_from(buf, 0)
    pos = 512
    nsub = max(int(fnsub), 1) if flags & TMULTI else 1
    common_x = None
    if not flags & TXYXYS:
        if flags & TXVALS:
            common_x = np.frombuffer(buf, "<f4", fnpts, pos).astype(np.float64)
            pos += 4 * fnpts
        else:
            common_x = np.linspace(ffirst, flast, fnpts)
    ys: List[np.ndarray] = []
    xs: List[np.ndarray] = []
    z = []
    for _ in range(nsub):
        if pos + 32 > len(buf):
            break
        (_sflags, subexp, _idx, subtime, _subnext, _noise, subnpts, _scan, _wlevel,
         _resv) = _SUB_HDR.unpack_from(buf, pos)
        pos += 32
        if flags & TXYXYS:
            npts = int(subnpts)
            x = np.frombuffer(buf, "<f4", npts, pos).astype(np.float64)
            pos += 4 * npts
        else:
            npts = int(fnpts)
            x = common_x
        exp = subexp if flags & TMULTI else fexp
        if not -128 <= exp <= 127:
            exp = fexp
        y, pos = _y_values(buf, pos, npts, exp, flags)
        xs.append(x)
        ys.append(y)
        z.append(float(subtime))

    meta = {"comment": _text(fcmnt), "source_instrument": _text(fsource), "format": "SPC (new)"}
    if flags & TALABS:
        meta["axis_labels"] = _text(fcatxt)
    if flogoff and 0 < flogoff < len(buf):
        meta["log"] = _read_log(buf, flogoff)
    return _make_record(xs, ys, z, X_UNITS.get(fxtype, ""), Y_UNITS.get(fytype, ""), meta,
                        z_label=X_UNITS.get(fztype, "z"))


def _read_log(buf: bytes, off: int) -> str:
    try:
        sizd, _sizm, txto, _binsz, _disksz = struct.unpack_from("<iiiii", buf, off)
        start = off + txto
        end = min(off + sizd, len(buf))
        return buf[start:end].split(b"\x00", 1)[0].decode("latin-1", errors="replace").strip()
    except struct.error:
        return ""


def _read_old(buf: bytes, path) -> Record:
    (flags, _ver, oexp, onpts, ofirst, olast, oxtype, oytype, *_rest) = _OLD_HDR.unpack_from(buf, 0)
    fcmnt = _OLD_HDR.unpack_from(buf, 0)[-2]
    npts = int(round(onpts))
    x = np.linspace(ofirst, olast, npts)
    pos = 224  # the first sub-file header follows the 224-byte main header
    ys, xs, z = [], [], []
    while pos + 32 + 4 * npts <= len(buf):
        (_sflags, subexp, _idx, subtime, *_r) = _SUB_HDR.unpack_from(buf, pos)
        pos += 32
        exp = subexp if 0 < subexp < 128 else oexp
        raw = np.frombuffer(buf, np.uint8, 4 * npts, pos).reshape(-1, 4)
        # old-format integers store the high-order 16-bit word first
        ints = raw[:, [2, 3, 0, 1]].copy().view("<i4").ravel()
        if exp == 128 or exp == -128:
            y = np.frombuffer(buf, "<f4", npts, pos).astype(np.float64)
        else:
            y = ints.astype(np.float64) * 2.0 ** (exp - 32)
        pos += 4 * npts
        xs.append(x)
        ys.append(y)
        z.append(float(subtime))
        if not flags & TMULTI:
            break
    meta = {"comment": _text(fcmnt), "format": "SPC (old)"}
    return _make_record(xs, ys, z, X_UNITS.get(oxtype, ""), Y_UNITS.get(oytype, ""), meta)


def _make_record(xs, ys, z, x_units, y_units, meta, z_label="z") -> Record:
    if not ys:
        raise ValueError("SPC file contains no spectra")
    same = all(len(x) == len(xs[0]) and np.allclose(x, xs[0]) for x in xs)
    if same:
        arrays = {INTENSITY: np.vstack(ys)}
        wn = xs[0]
    else:
        arrays = {INTENSITY: ys}
        wn = xs
    positions = None
    names = []
    if len(ys) > 1:
        positions = np.asarray(z, dtype=np.float64)[:, None]
    return Record(arrays, wn, names or None, positions, ["z"] if positions is not None else [],
                  None, x_units, y_units, meta)


def write_spc(path, record: Record, x_type: int = 13, y_type: int = 0, comment: str = "") -> Path:
    """Write the single array of ``record`` as a new-format ``.spc`` file.

    One sub-file per spectrum, float Y. An evenly spaced axis is stored as
    first/last values, any other axis as an explicit X array; spectra of
    different lengths use one X array per sub-file (``TXYXYS``).
    """
    if len(record.arrays) != 1:
        raise ValueError("write_spc expects a record with one array")
    ys = next(iter(record.arrays.values()))
    ys = [np.asarray(y, dtype=np.float64).ravel() for y in ys]
    n = len(ys)
    if isinstance(record.wavenumbers, list):
        xs = [np.asarray(x, dtype=np.float64) for x in record.wavenumbers]
    elif record.wavenumbers is not None:
        xs = [np.asarray(record.wavenumbers, dtype=np.float64)] * n
    else:
        xs = [np.arange(len(y), dtype=np.float64) for y in ys]
    common = all(len(x) == len(xs[0]) and np.array_equal(x, xs[0]) for x in xs)
    flags = TMULTI if n > 1 else 0
    if not common:
        flags |= TXYXYS | TXVALS | TMULTI
    x0 = xs[0]
    even = common and len(x0) > 1 and np.allclose(np.diff(x0), (x0[-1] - x0[0]) / (len(x0) - 1),
                                                  rtol=1e-6, atol=1e-9)
    if common and not even:
        flags |= TXVALS
    zpos = None
    if record.positions is not None and record.positions.shape[0] == n:
        zpos = record.positions[:, 0]
    if n > 1:
        flags |= TORDRD if zpos is not None else 0

    fnpts = len(x0) if common else 0
    header = _NEW_HDR.pack(
        flags, 0x4B, 0, -128, fnpts, float(x0[0]), float(x0[-1]), n, x_type, y_type, 0, 0, 0,
        b"", b"RADAR", 0, b"\x00" * 32, comment.encode("latin-1", "replace")[:129], b"", 0, 0, 0,
        0, 0, 0.0, b"", 0.0, 0, 0.0, 0, b"",
    )
    parts = [header]
    if common and not even:
        parts.append(x0.astype("<f4").tobytes())
    for i, (x, y) in enumerate(zip(xs, ys)):
        t = float(zpos[i]) if zpos is not None else float(i)
        parts.append(_SUB_HDR.pack(0, -128, i, t, 0.0, 0.0, len(y) if not common else 0, 0, 0.0, b""))
        if not common:
            parts.append(x.astype("<f4").tobytes())
        parts.append(y.astype("<f4").tobytes())
    path = Path(path)
    path.write_bytes(b"".join(parts))
    return path
