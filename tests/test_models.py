# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
import numpy as np
import pytest

import radar_raman
from radar_raman._weights import is_legacy_weight_file, load_legacy_weights, pretrained_weights_path
from radar_raman.models import COMPONENTS, LEGACY_LAYER_NAMES, build_model


@pytest.mark.parametrize("arch,n_params", [("standard", 585_299), ("ensemble", 280_592)])
def test_parameter_counts(arch, n_params):
    assert build_model(arch).count_params() == n_params


@pytest.mark.parametrize("arch", ["standard", "ensemble"])
@pytest.mark.parametrize("length", [64, 333, 1024])
def test_output_shapes_any_length(arch, length):
    m = build_model(arch)
    out = m.predict(np.random.default_rng(0).random((3, length, 1)), verbose=0)
    assert len(out) == 4
    for o in out:
        assert o.shape == (3, length)


def test_unknown_architecture():
    with pytest.raises(ValueError):
        build_model("unet")


def test_dim_red_must_be_square():
    with pytest.raises(ValueError):
        build_model("standard", dim_red_size=200)


def test_seeded_build_is_deterministic():
    a = build_model("ensemble", seed=3).get_weights()
    b = build_model("ensemble", seed=3).get_weights()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("arch", ["standard", "ensemble"])
def test_bundled_weights_exist(arch):
    p = pretrained_weights_path(arch)
    assert p.exists() and is_legacy_weight_file(p)


@pytest.mark.parametrize("arch", ["standard", "ensemble"])
@pytest.mark.parametrize("key", ["X1024", "X1738"])
def test_matches_original_implementation(pretrained, reference, arch, key):
    """The Keras 3 port + bundled weights reproduce the original code's outputs."""
    out = pretrained[arch].predict_normalized(reference[key])
    for c in COMPONENTS:
        np.testing.assert_allclose(out[c], reference[f"{arch}_{key}_{c}"], atol=1e-5, rtol=0)


def test_legacy_loader_rejects_wrong_architecture():
    m = build_model("standard")
    with pytest.raises(ValueError):
        load_legacy_weights(m, pretrained_weights_path("ensemble"), "standard")


def test_legacy_mapping_covers_all_weighted_layers():
    for arch, mapping in LEGACY_LAYER_NAMES.items():
        weighted = {l.name for l in build_model(arch).layers if l.weights}
        assert weighted == set(mapping)


def test_saved_models_record_provenance(tmp_path):
    import json
    import zipfile

    import keras

    from radar_raman import _provenance

    m = radar_raman.load_model("standard")
    k = m.save(tmp_path / "m.keras")
    w = m.save(tmp_path / "m.weights.h5")
    info = json.loads(zipfile.ZipFile(k).read("radar_info.json"))
    assert info["saved_with"]["keras"] == keras.__version__
    assert info["saved_with"]["radar"] == radar_raman.__version__
    assert "tensorflow" in info["saved_with"] and info["architecture"] == "standard"
    assert info["hyperparameters"]["scaler"] == 20
    assert info["training"]["spectra_seen"] == 240_000_000
    # weights-only files need no architecture argument any more
    m2 = radar_raman.load_model(w)
    assert m2.architecture == "standard"
    assert radar_raman.model_info(w)["compatibility"] == []
    assert radar_raman.model_info(k)["saved_with"]["keras"] == keras.__version__
    # a model trained further records its training and where it started from
    m.training_info = None
    assert "training" not in _provenance.make_info(m)


def test_provenance_flags_newer_software(tmp_path):
    import json

    import h5py
    import pytest as _pytest

    m = radar_raman.load_model("standard")
    w = m.save(tmp_path / "m.weights.h5")
    with h5py.File(w, "a") as f:
        info = json.loads(f.attrs["radar_info"])
        info["saved_with"]["keras"] = "9.0.0"
        f.attrs["radar_info"] = json.dumps(info)
    assert any("keras 9.0.0" in n for n in radar_raman.model_info(w)["compatibility"])
    with _pytest.warns(UserWarning, match="keras 9.0.0"):
        radar_raman.load_model(w)


def test_provenance_of_foreign_files(tmp_path):
    from radar_raman._weights import pretrained_weights_path

    info = radar_raman.model_info(pretrained_weights_path("ensemble"))
    assert info["saved_with"]["keras"] == "2.7.0"
    assert radar_raman.model_info("ensemble")["compatibility"] == []
    plain = radar_raman.build_model("standard")
    plain.save(tmp_path / "plain.keras")        # saved by Keras directly, no RADAR record
    info = radar_raman.model_info(tmp_path / "plain.keras")
    assert info["saved_with"]["keras"] and "note" in info
