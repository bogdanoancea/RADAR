# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
import json

import h5py
import numpy as np
import pytest

import radar_raman
from radar_raman.io import load_components, read_spectra, save_components, save_spectra


@pytest.fixture
def spectra():
    rng = np.random.default_rng(0)
    return rng.random((3, 50)), np.linspace(200, 1800, 50)


def test_two_column_text(tmp_path):
    wn = np.linspace(100, 2000, 40)
    y = np.sin(wn / 100) + 2
    p = tmp_path / "s.txt"
    np.savetxt(p, np.column_stack([wn, y]), header="raman shift  intensity")
    d = read_spectra(p)
    assert d.intensities.shape == (1, 40)
    np.testing.assert_allclose(d.wavenumbers, wn)
    np.testing.assert_allclose(d.intensities[0], y)


def test_csv_columns_with_header_roundtrip(tmp_path, spectra):
    X, wn = spectra
    p = save_spectra(tmp_path / "s.csv", X, wavenumbers=wn, names=["a", "b", "c"])
    d = read_spectra(p)
    np.testing.assert_allclose(d.intensities, X)
    np.testing.assert_allclose(d.wavenumbers, wn)
    assert d.names == ["a", "b", "c"]


def test_semicolon_and_descending_axis(tmp_path):
    wn = np.linspace(1800, 200, 30)
    X = np.random.default_rng(1).random((2, 30))
    p = tmp_path / "s.csv"
    with open(p, "w") as f:
        f.write("# exported spectrum\nshift;s1;s2\n")
        for i in range(30):
            f.write(f"{wn[i]};{X[0, i]};{X[1, i]}\n")
    d = read_spectra(p)
    np.testing.assert_allclose(d.wavenumbers, wn)
    np.testing.assert_allclose(d.intensities, X)


def test_rows_layout(tmp_path, spectra):
    X, wn = spectra
    p = tmp_path / "rows.csv"
    np.savetxt(p, np.vstack([wn, X]), delimiter=",")
    d = read_spectra(p, layout="rows")
    np.testing.assert_allclose(d.wavenumbers, wn)
    np.testing.assert_allclose(d.intensities, X)
    np.savetxt(p, X, delimiter=",")
    d = read_spectra(p)  # auto -> rows (more columns than rows)
    assert d.intensities.shape == (3, 50) and d.wavenumbers is None


def test_npy_npz(tmp_path, spectra):
    X, wn = spectra
    np.save(tmp_path / "x.npy", X[0])
    assert read_spectra(tmp_path / "x.npy").intensities.shape == (1, 50)
    save_spectra(tmp_path / "x.npz", X, wavenumbers=wn)
    d = read_spectra(tmp_path / "x.npz")
    np.testing.assert_allclose(d.intensities, X)
    np.testing.assert_allclose(d.wavenumbers, wn)


def test_h5_variants(tmp_path, spectra):
    X, wn = spectra
    p = tmp_path / "a.h5"
    with h5py.File(p, "w") as f:
        f["my/data"] = X
        f["wavenumbers"] = wn
    d = read_spectra(p, key="my/data")
    np.testing.assert_allclose(d.intensities, X)
    np.testing.assert_allclose(d.wavenumbers, wn)
    # live-cell layout: Regions/<r>/Dataset with a map of spectra
    p2 = tmp_path / "cells.h5"
    with h5py.File(p2, "w") as f:
        f["Regions/r1/Dataset"] = np.ones((2, 3, 50))
        f["Regions/r2/Dataset"] = np.zeros((4, 50))
    assert read_spectra(p2).intensities.shape == (10, 50)


def test_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_spectra(tmp_path / "nope.csv")
    (tmp_path / "x.xyz").write_text("1 2 3")
    with pytest.raises(ValueError):
        read_spectra(tmp_path / "x.xyz")


def test_spectraldata_into_decompose(tmp_path, pretrained, spectra):
    X, wn = spectra
    p = save_spectra(tmp_path / "s.csv", X, wavenumbers=wn)
    c = pretrained["ensemble"].decompose(read_spectra(p))
    np.testing.assert_allclose(c.wavenumbers, wn)
    assert c.spectra.shape == (3, 50)


@pytest.mark.parametrize("ext", [".npz", ".h5"])
def test_components_roundtrip(tmp_path, pretrained, spectra, ext):
    X, wn = spectra
    c = pretrained["standard"].decompose(X, wavenumbers=wn)
    p = save_components(c, tmp_path / f"out{ext}")
    c2 = load_components(p)
    for k in ("spectra", "baseline", "cosmic_rays", "noise", "peaks", "wavenumbers"):
        np.testing.assert_allclose(getattr(c2, k), getattr(c, k))
    assert c2.model == "standard" and c2.units == "original"


def test_components_csv(tmp_path, pretrained, spectra):
    X, wn = spectra
    many = pretrained["standard"].decompose(X, wavenumbers=wn)
    p = many.save(tmp_path / "many.csv")
    table = np.genfromtxt(p, delimiter=",", names=True)
    # one tidy table: spectrum id, wavenumber, then one column per array
    assert table.dtype.names == ("spectrum", "wavenumber", "spectra", "baseline", "cosmic_rays",
                                 "noise", "peaks")
    np.testing.assert_allclose(table["peaks"][table["spectrum"] == 2], many.peaks[1])
    back = radar_raman.load_components(p)
    np.testing.assert_allclose(back.peaks, many.peaks)


def test_dataset_roundtrip(tmp_path):
    ds = radar_raman.generate_dataset(5, length=100, seed=0)
    p = ds.save(tmp_path / "ds")
    assert p.suffix == ".npz"
    ds2 = radar_raman.SyntheticDataset.load(p)
    np.testing.assert_array_equal(ds.X, ds2.X)
    np.testing.assert_array_equal(ds.peaks, ds2.peaks)
    assert ds2.params["length"] == 100
