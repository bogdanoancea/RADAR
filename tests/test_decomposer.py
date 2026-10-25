# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
import numpy as np
import pytest

import radar
from radar import Components, Decomposer


@pytest.mark.parametrize("arch", ["standard", "ensemble"])
def test_decompose_2d(pretrained, small_dataset, arch):
    c = pretrained[arch].decompose(small_dataset.X)
    assert isinstance(c, Components)
    assert c.baseline.shape == small_dataset.X.shape and len(c) == 16
    assert c.model == arch and c.units == "original"
    assert np.all(c.peaks >= 0) and np.all(c.cosmic_rays >= 0)


def test_decompose_1d(pretrained, small_dataset):
    c = pretrained["ensemble"].decompose(small_dataset.X[0])
    assert c.spectra.ndim == 1 and c.peaks.shape == (512,) and c.n_spectra == 1


def test_scale_invariance_and_original_units(pretrained, small_dataset):
    m = pretrained["standard"]
    x = small_dataset.X[:4]
    a = m.decompose(x)
    b = m.decompose(x * 1234.5 + 300.0)
    # identical up to float32 rounding of the normalised input
    tol = 1e-4 * 1234.5
    np.testing.assert_allclose(b.peaks, a.peaks * 1234.5, atol=tol)
    np.testing.assert_allclose(b.baseline, a.baseline * 1234.5 + 300.0, atol=tol)
    # standard model: noise is the remainder, so components sum to the input
    np.testing.assert_allclose(b.reconstruction, b.spectra, atol=tol)


def test_normalized_units(pretrained, small_dataset):
    c = pretrained["ensemble"].decompose(small_dataset.X[:2] * 10, units="normalized")
    assert c.units == "normalized"
    assert np.isclose(c.spectra.max(), 1) and np.isclose(c.spectra.min(), 0)


def test_ragged_input(pretrained, small_dataset):
    x = small_dataset.X
    res = pretrained["ensemble"].decompose([x[0], x[1][:300], x[2]])
    assert isinstance(res, list) and [r.length for r in res] == [512, 300, 512]
    same = pretrained["ensemble"].decompose([x[0], x[1]])
    assert isinstance(same, Components) and same.spectra.shape == (2, 512)


def test_wavenumbers_and_validation(pretrained, small_dataset):
    wn = np.linspace(400, 1800, 512)
    c = pretrained["ensemble"].decompose(small_dataset.X[0], wavenumbers=wn)
    np.testing.assert_array_equal(c.wavenumbers, wn)
    with pytest.raises(ValueError):
        pretrained["ensemble"].decompose(small_dataset.X[0], wavenumbers=wn[:10])
    bad = small_dataset.X[0].copy()
    bad[3] = np.nan
    with pytest.raises(ValueError):
        pretrained["ensemble"].decompose(bad)
    with pytest.raises(ValueError):
        pretrained["ensemble"].decompose(np.zeros((1,)))  # a single point


def test_decompose_map_cube(pretrained, small_dataset):
    cube = small_dataset.X[:6].reshape(2, 3, -1)
    c = pretrained["standard"].decompose(cube)
    assert c.map_shape == (2, 3) and c.spectra.shape == (6, 512)
    assert c.as_map("peaks").shape == (2, 3, 512)
    wn = np.linspace(400, 1800, 512)
    c2 = pretrained["standard"].decompose(cube, wavenumbers=wn)
    assert c2.band_image(900, 1100).shape == (2, 3)


def test_flat_spectrum_does_not_crash(pretrained):
    c = pretrained["standard"].decompose(np.full(200, 5.0))
    assert np.all(np.isfinite(c.peaks))


def test_derived_spectra(pretrained, small_dataset):
    c = pretrained["standard"].decompose(small_dataset.X[:2])
    np.testing.assert_allclose(c.denoised, c.spectra - c.noise)
    np.testing.assert_allclose(c.baseline_corrected, c.spectra - c.baseline)
    np.testing.assert_allclose(c.despiked, c.spectra - c.cosmic_rays)
    assert c.corrected is c.peaks
    assert c[0].spectra.shape == (512,)


def test_module_level_decompose(small_dataset):
    c = radar.decompose(small_dataset.X[:2], model="standard")
    assert c.model == "standard"


def test_evaluate_quality(pretrained):
    """Pre-trained models recover the components of original-style test data."""
    ds = radar.generate_dataset(64, length=1024, seed=2024)
    for arch, dec in pretrained.items():
        res = dec.evaluate(ds)
        assert res["baseline"]["r2"] > 0.85, (arch, res)
        assert res["noise"]["r2"] > 0.85, (arch, res)
        assert res["peaks"]["r2"] > 0.7, (arch, res)
        assert 0 < res["peaks"]["max_error"] < 1, (arch, res)
        stats = dec.evaluate(ds, statistics=True)
        assert stats["peaks"]["max_error_mean"] >= stats["peaks"]["rmse_mean"]


@pytest.mark.parametrize("arch", ["standard", "ensemble"])
def test_save_load_keras(pretrained, small_dataset, tmp_path, arch):
    m = pretrained[arch]
    p = m.save(tmp_path / "model.keras")
    m2 = radar.load_model(p)
    assert m2.architecture == arch
    np.testing.assert_allclose(
        m2.decompose(small_dataset.X[:3]).peaks, m.decompose(small_dataset.X[:3]).peaks, atol=1e-6
    )


def test_save_load_weights(pretrained, small_dataset, tmp_path):
    m = pretrained["ensemble"]
    p = m.save(tmp_path / "model.weights.h5")
    assert radar.load_model(p).architecture == "ensemble"  # read from the file's record
    m2 = radar.load_model(p, architecture="ensemble")
    np.testing.assert_allclose(
        m2.decompose(small_dataset.X[:3]).noise, m.decompose(small_dataset.X[:3]).noise, atol=1e-6
    )


def test_load_original_h5(tmp_path):
    from radar._weights import pretrained_weights_path

    m = radar.load_model(pretrained_weights_path("standard"), architecture="standard")
    assert m.architecture == "standard"


def test_save_bad_name(pretrained, tmp_path):
    with pytest.raises(ValueError):
        pretrained["standard"].save(tmp_path / "model.pkl")
    with pytest.raises(FileNotFoundError):
        radar.load_model(tmp_path / "missing.keras")


@pytest.mark.slow
@pytest.mark.parametrize("arch", ["standard", "ensemble"])
def test_train_synthetic_stream(arch, tmp_path):
    dec = Decomposer.new(arch)
    w0 = [w.copy() for w in dec.model.get_weights()]
    hist = dec.train(
        epochs=1, steps_per_epoch=2, batch_size=4, length=128,
        validation_size=4, validation_length=128, verbose=0,
        checkpoint_path=str(tmp_path / "best.keras"),
    )
    assert "loss" in hist.history and "val_loss" in hist.history
    assert any(not np.array_equal(a, b) for a, b in zip(w0, dec.model.get_weights()))
    assert (tmp_path / "best.keras").exists()
    c = dec.decompose(np.random.default_rng(0).random(128))
    assert c.peaks.shape == (128,)


@pytest.mark.slow
def test_train_fixed_dataset_and_finetune():
    tr, te = radar.train_test_datasets(8, 4, length=128, seed=0)
    dec = Decomposer.pretrained("standard")
    hist = dec.train(tr, epochs=1, batch_size=4, learning_rate=1e-5,
                     validation_data=te, verbose=0)
    assert len(hist.history["loss"]) == 1
    dec2 = Decomposer.new("ensemble")
    dec2.train((tr.X, {c: getattr(tr, c) for c in radar.COMPONENTS}), epochs=1,
               batch_size=4, validation_data=None, verbose=0)


def test_keras_file_from_newer_keras_falls_back_to_weights(pretrained, small_dataset, tmp_path, monkeypatch):
    """A .keras file whose config the installed Keras cannot read (e.g. written
    by a newer Keras) is loaded by rebuilding the architecture + weights."""
    import keras

    m = pretrained["ensemble"]
    p = m.save(tmp_path / "model.keras")

    def fail(*args, **kwargs):
        raise TypeError("GlorotUniform.__init__() got an unexpected keyword argument 'input_axes'")

    monkeypatch.setattr(keras.models, "load_model", fail)
    with pytest.warns(RuntimeWarning, match="newer Keras"):
        m2 = radar.load_model(p)
    assert m2.architecture == "ensemble"
    np.testing.assert_allclose(
        m2.decompose(small_dataset.X[:3]).peaks, m.decompose(small_dataset.X[:3]).peaks, atol=1e-6
    )


@pytest.mark.slow
def test_training_is_recorded(tmp_path):
    import json
    import zipfile

    dec = radar.load_model("standard")
    tr, _ = radar.train_test_datasets(8, 2, length=64, seed=0)
    dec.train(tr, epochs=1, batch_size=4, learning_rate=1e-6, validation_data=None, verbose=0)
    info = json.loads(zipfile.ZipFile(dec.save(tmp_path / "t.keras")).read("radar_info.json"))
    assert info["training"]["spectra_seen"] == 8
    assert info["training"]["started_from"]["spectra_seen"] == 240_000_000
    back = radar.load_model(tmp_path / "t.keras")
    assert back.training_info["spectra_seen"] == 8
