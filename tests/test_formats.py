# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""File formats: round trips, instrument text dialects and (optionally) real
instrument files.

Real vendor sample files are not distributed with RADAR. To run those tests,
point ``RADAR_SAMPLE_DATA`` to a folder containing the test data of
https://github.com/rohanisaac/spc (``spc/test_data``),
https://github.com/nzhagen/jcamp (``jcamp/data``) and
https://github.com/hyperspy/rosettasciio (``rosettasciio/rsciio/tests/data``).
"""

import glob
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

import radar_raman
from radar_raman.io import Record

def _works(module):
    """True if the module is installed AND imports (a broken install counts as missing)."""
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


HAS = {m: _works(m) for m in ("pandas", "pyarrow", "openpyxl", "renishawWiRE", "rsciio")}
SAMPLES = os.environ.get("RADAR_SAMPLE_DATA")


@pytest.fixture
def spec():
    rng = np.random.default_rng(0)
    wn = np.linspace(200.0, 1800.0, 64)
    X = rng.random((3, 64)) * 1000
    return X, wn, ["a", "b", "c"]


WRITABLE = ["csv", "tsv", "txt", "dat", "h5", "nc", "npz", "mat", "json", "db", "spc", "jdx"]
OPTIONAL = {"xlsx": ("pandas", "openpyxl"), "parquet": ("pandas", "pyarrow"), "feather": ("pandas", "pyarrow")}


def _available(ext):
    return all(HAS[m] for m in OPTIONAL.get(ext, ()))


ALL_EXT = WRITABLE + list(OPTIONAL)


@pytest.mark.parametrize("ext", ALL_EXT)
def test_spectra_roundtrip(tmp_path, spec, ext):
    if not _available(ext):
        pytest.skip(f"{ext} needs {OPTIONAL[ext]}")
    X, wn, names = spec
    p = radar_raman.write(X, tmp_path / f"s.{ext}", wavenumbers=wn, names=names)
    d = radar_raman.read(p)
    tol = 1e-4 if ext == "spc" else 1e-9  # SPC stores float32
    np.testing.assert_allclose(d.intensities, X, rtol=tol)
    np.testing.assert_allclose(d.wavenumbers, wn, rtol=1e-6)
    if ext not in ("spc", "npy"):
        assert d.names == names


@pytest.mark.parametrize("ext", ALL_EXT)
def test_components_roundtrip(tmp_path, pretrained, spec, ext):
    if not _available(ext):
        pytest.skip(f"{ext} needs {OPTIONAL[ext]}")
    X, wn, names = spec
    c = pretrained["standard"].decompose(radar_raman.SpectralData.from_arrays(X, wn, names))
    out = c.save(tmp_path / f"c.{ext}")
    first = out[0] if isinstance(out, list) else out
    c2 = radar_raman.load_components(first)
    tol = 1e-4 if ext == "spc" else 1e-9
    for k in ("spectra", "baseline", "cosmic_rays", "noise", "peaks"):
        np.testing.assert_allclose(getattr(c2, k), getattr(c, k), rtol=tol, atol=tol * 1000)
    np.testing.assert_allclose(c2.wavenumbers, wn, rtol=1e-6)


@pytest.mark.parametrize("ext", ["npz", "h5", "json", "db", "csv", "mat"])
def test_dataset_roundtrip_formats(tmp_path, ext):
    ds = radar_raman.generate(4, length=50, seed=1)
    p = ds.save(tmp_path / f"d.{ext}")
    ds2 = radar_raman.load_dataset(p)
    np.testing.assert_allclose(ds2.X, ds.X, atol=1e-6)
    np.testing.assert_allclose(ds2.peaks, ds.peaks, atol=1e-6)


@pytest.mark.parametrize("ext", ["npz", "h5", "json", "db", "csv", "spc", "jdx"])
def test_ragged_roundtrip(tmp_path, ext):
    rng = np.random.default_rng(2)
    ys = [rng.random(40), rng.random(55)]
    xs = [np.linspace(100, 900, 40), np.linspace(150, 1000, 55)]
    rec = Record({"intensity": ys}, xs, ["p", "q"])
    p = radar_raman.write(rec, tmp_path / f"r.{ext}")
    d = radar_raman.read(p)
    assert d.ragged
    for a, b in zip(d.intensities, ys):
        np.testing.assert_allclose(a, b, rtol=1e-6)
    for a, b in zip(d.wavenumbers, xs):
        np.testing.assert_allclose(a, b, rtol=1e-6)


def test_map_positions_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    cube = rng.random((2, 3, 30))
    d = radar_raman.SpectralData.from_arrays(cube, np.arange(30.0))
    assert d.map_shape == (2, 3)
    yy, xx = np.meshgrid([0.0, 1.0], [0.0, 1.0, 2.0], indexing="ij")
    d.positions, d.position_names = np.column_stack([xx.ravel(), yy.ravel()]), ["x", "y"]
    for ext in ("csv", "db", "h5", "json"):
        p = radar_raman.write(d, tmp_path / f"m.{ext}")
        back = radar_raman.read(p)
        assert back.map_shape == (2, 3), ext
        np.testing.assert_allclose(back.as_map(), cube)


# ---------------------------------------------------------------------------
# instrument text dialects
# ---------------------------------------------------------------------------
def test_renishaw_wire_text_map(tmp_path):
    wn = np.linspace(1800, 200, 20)  # WiRE exports run high -> low
    lines = ["#X\t#Y\t#Wave\t#Intensity"]
    vals = {}
    for y in (0.0, 1.0):
        for x in (0.0, 1.0, 2.0):
            ints = np.round(np.random.default_rng(int(10 * x + y)).random(20) * 100, 3)
            vals[(x, y)] = ints
            lines += [f"{x}\t{y}\t{w}\t{i}" for w, i in zip(wn, ints)]
    p = tmp_path / "wire_map.txt"
    p.write_text("\n".join(lines))
    d = radar_raman.read(p)
    assert len(d) == 6 and d.map_shape == (2, 3)
    np.testing.assert_allclose(d.wavenumbers, wn)
    np.testing.assert_allclose(d.intensities[0], vals[(0.0, 0.0)])


def test_renishaw_wire_text_headerless(tmp_path):
    wn = np.linspace(1500, 300, 15)
    rows = [f"{z}\t{w}\t{z * 10 + j}" for z in (0.0, 5.0) for j, w in enumerate(wn)]
    p = tmp_path / "zscan.txt"
    p.write_text("\n".join(rows))
    d = radar_raman.read(p)
    assert len(d) == 2 and d.length == 15
    np.testing.assert_allclose(d.positions[:, 0], [0.0, 5.0])


def test_labspec_text_map(tmp_path):
    wn = np.linspace(100, 1000, 10)
    lines = ["#Acq. time (s)=1", "#Accumulations=2", "\t\t" + "\t".join(map(str, wn))]
    for y in (0, 1):
        for x in (0, 1):
            lines.append(f"{x}\t{y}\t" + "\t".join(str(x + y + k) for k in range(10)))
    p = tmp_path / "labspec_map.txt"
    p.write_text("\n".join(lines))
    d = radar_raman.read(p)
    assert len(d) == 4
    np.testing.assert_allclose(d.wavenumbers, wn)
    assert d.positions.shape == (4, 2) and d.map_shape == (2, 2)


def test_ocean_insight_text(tmp_path):
    body = "\n".join(f"{500 + i * 0.5}\t{100 + i}" for i in range(30))
    p = tmp_path / "ocean.txt"
    p.write_text("Data from spectrometer.txt Node\n\nDate: Mon Oct 06\nUser: lab\n"
                 "Integration Time (sec): 1.0\n>>>>>Begin Spectral Data<<<<<\n" + body +
                 "\n>>>>>End Spectral Data<<<<<\n")
    d = radar_raman.read(p)
    assert d.intensities.shape == (1, 30) and d.wavenumbers[0] == 500


def test_rruff_text(tmp_path):
    body = "\n".join(f"{150 + i}, {i * 2.5}" for i in range(25))
    p = tmp_path / "mineral.txt"
    p.write_text("##NAMES=Quartz\n##RRUFFID=R040031\n##LOCALITY=Somewhere\n" + body + "\n##END=\n")
    d = radar_raman.read(p)
    assert d.intensities.shape == (1, 25) and d.wavenumbers[-1] == 174


def test_decimal_comma(tmp_path):
    p = tmp_path / "eu.csv"
    p.write_text("Raman shift;Intensity\n" + "\n".join(f"{100 + i},5;{i},25" for i in range(12)))
    d = radar_raman.read(p)
    np.testing.assert_allclose(d.wavenumbers[:2], [100.5, 101.5])
    np.testing.assert_allclose(d.intensities[0, :2], [0.25, 1.25])


def test_sqlite_custom_query(tmp_path, spec):
    X, wn, names = spec
    p = radar_raman.write(X, tmp_path / "s.db", wavenumbers=wn, names=names)
    d = radar_raman.read(p, query="SELECT spectrum, wavenumber, intensity FROM spectra WHERE spectrum = 2")
    np.testing.assert_allclose(d.intensities[0], X[1])


def test_jcamp_asdf_compressed(tmp_path):
    # line 1: SQZ A0 = 10, DIF J = +1 -> 11, DUP T repeats the last token (J) -> 12
    # line 2: A2 = 12 is the DIF Y-check (dropped), J -> 13, % (= +0) -> 13
    p = tmp_path / "difdup.jdx"
    p.write_text("##TITLE=test\n##JCAMP-DX=4.24\n##XUNITS=1/CM\n##YUNITS=COUNTS\n##XFACTOR=1\n"
                 "##YFACTOR=0.5\n##FIRSTX=100\n##LASTX=104\n##NPOINTS=5\n"
                 "##XYDATA=(X++(Y..Y))\n100A0JT\n102A2J%\n##END=\n")
    d = radar_raman.read(p)
    np.testing.assert_allclose(d.intensities[0], np.array([10, 11, 12, 13, 13]) * 0.5)
    np.testing.assert_allclose(d.wavenumbers, [100, 101, 102, 103, 104])


def test_unknown_extension(tmp_path):
    (tmp_path / "x.unknown").write_text("1 2")
    with pytest.raises(ValueError):
        radar_raman.read(tmp_path / "x.unknown")
    d = radar_raman.read(tmp_path / "x.unknown", format="txt")
    assert d.intensities.size == 2


def test_formats_listing():
    rows = radar_raman.formats()
    names = {r["format"] for r in rows}
    assert {"csv", "spc", "jcamp", "wdf", "sqlite", "parquet", "hdf5"} <= names


def test_decompose_file_and_output(tmp_path, spec):
    X, wn, names = spec
    p = radar_raman.write(X, tmp_path / "in.spc", wavenumbers=wn)
    c = radar_raman.decompose(p, model="standard", output=tmp_path / "out.h5")
    assert c.spectra.shape == (3, 64)
    np.testing.assert_allclose(radar_raman.load_components(tmp_path / "out.h5").peaks, c.peaks)


# ---------------------------------------------------------------------------
# real instrument files (optional)
# ---------------------------------------------------------------------------
def _samples(pattern):
    if not SAMPLES:
        return []
    return sorted(glob.glob(os.path.join(SAMPLES, pattern), recursive=True))


@pytest.mark.skipif(not SAMPLES, reason="set RADAR_SAMPLE_DATA to run")
@pytest.mark.parametrize("pattern,minimum", [
    ("spc/test_data/*.[sS][pP][cC]", 27),
    ("jcamp/data/raman_spectra/*.jdx", 1),
    ("rosettasciio/rsciio/tests/data/renishaw/*.wdf", 14),
    ("rosettasciio/rsciio/tests/data/jobinyvon/*.xml", 9),
    ("rosettasciio/rsciio/tests/data/trivista/*.tvf", 9),
])
def test_real_files(pattern, minimum):
    files = _samples(pattern)
    if "wdf" in pattern and not (HAS["renishawWiRE"] or HAS["rsciio"]):
        pytest.skip("needs renishawWiRE")
    if ("xml" in pattern or "tvf" in pattern) and not HAS["rsciio"]:
        pytest.skip("needs rosettasciio")
    assert len(files) >= minimum
    for f in files:
        d = radar_raman.read(f)
        assert len(d) >= 1, f


def test_sqlite_append_and_fail(tmp_path, spec):
    X, wn, names = spec
    p = tmp_path / "s.db"
    radar_raman.write(X[:2], p, wavenumbers=wn)
    radar_raman.write(X[2:], p, wavenumbers=wn, table="more", if_exists="append")
    np.testing.assert_allclose(radar_raman.read(p, table="more").intensities, X[2:])
    np.testing.assert_allclose(radar_raman.read(p).intensities, X[:2])  # first table kept
    with pytest.raises(FileExistsError):
        radar_raman.write(X, p, wavenumbers=wn, if_exists="fail")
