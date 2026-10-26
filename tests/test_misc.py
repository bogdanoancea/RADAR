# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
import numpy as np
import pytest

import radar_raman
from radar_raman import cli, metrics
from radar_raman.preprocessing import denormalize_components, minmax_normalize
from radar_raman.training import ORIGINAL_LR_VALUES, original_learning_rate_schedule


def test_minmax_and_inverse():
    x = np.array([[1.0, 3.0, 5.0], [2.0, 2.0, 2.0]])
    xn, s = minmax_normalize(x)
    np.testing.assert_allclose(xn[0], [0, 0.5, 1])
    np.testing.assert_allclose(xn[1], 0)  # flat spectrum -> zeros, no NaN
    comps = {"baseline": xn * 0.5, "peaks": xn * 0.5}
    back = denormalize_components(comps, s)
    np.testing.assert_allclose(back["baseline"] + back["peaks"], x)
    with pytest.raises(ValueError):
        minmax_normalize(np.zeros(3))


def test_metrics():
    t = np.array([[0.0, 1.0, 2.0]])
    p = np.array([[0.5, 1.0, 1.0]])
    assert metrics.rmse(t, t)[0] == 0
    assert np.isclose(metrics.max_error(t, p)[0], 1.0)
    assert np.isclose(metrics.max_error(p, t)[0], 1.0)  # symmetric
    assert np.isclose(metrics.r2(t, t)[0], 1)
    assert np.isnan(metrics.r2(np.ones((1, 3)), np.ones((1, 3)))[0])
    assert metrics.snr(np.ones(10), np.zeros(10)) > 1e3


def test_schedule_matches_original():
    sched = original_learning_rate_schedule(2_400_000)
    assert [int(b) for b in sched.boundaries] == [200_000 * k for k in range(1, 12)]
    assert np.isclose(float(sched(0)), ORIGINAL_LR_VALUES[0])
    assert np.isclose(float(sched(2_399_999)), ORIGINAL_LR_VALUES[-1])
    assert original_learning_rate_schedule(5) == ORIGINAL_LR_VALUES[0]


def test_citation():
    assert "10.1002/adom.202500736" in radar_raman.citation()


def test_plotting(pretrained, small_dataset):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from radar_raman.plotting import plot_comparison

    c = pretrained["ensemble"].decompose(small_dataset.X[:2], units="normalized")
    assert len(c.plot(index=1).axes) == 5
    assert len(plot_comparison(small_dataset, c).axes) == 5


def test_cli_generate_decompose_evaluate(tmp_path, capsys):
    ds_path = tmp_path / "ds.npz"
    cli.main(["generate", "-n", "4", "--length", "128", "-o", str(ds_path)])
    assert ds_path.exists()
    X = radar_raman.SyntheticDataset.load(ds_path).X
    inp = radar_raman.save_spectra(tmp_path / "in.csv", X, wavenumbers=np.linspace(400, 1600, 128))
    out = tmp_path / "out.h5"
    cli.main(["decompose", str(inp), "-o", str(out), "-m", "standard"])
    c = radar_raman.load_components(out)
    assert c.spectra.shape == (4, 128) and c.wavenumbers is not None
    cli.main(["evaluate", str(ds_path), "-m", "ensemble"])
    assert '"peaks"' in capsys.readouterr().out
    cli.main(["info"])


@pytest.mark.slow
def test_cli_train(tmp_path):
    out = tmp_path / "m.keras"
    cli.main(["train", "-o", str(out), "--epochs", "1", "--steps", "2", "--batch-size", "4",
              "--length", "128", "--validation-size", "0", "--verbose", "0"])
    m = radar_raman.load_model(out)
    assert m.architecture == "ensemble"


def test_max_error_matches_sklearn_definition():
    rng = np.random.default_rng(0)
    t, p = rng.normal(size=(5, 40)), rng.normal(size=(5, 40))
    expected = [np.max(np.abs(a - b)) for a, b in zip(t, p)]  # sklearn.metrics.max_error per spectrum
    np.testing.assert_allclose(metrics.max_error(t, p), expected)


def test_error_statistics_and_evaluate(small_dataset):
    truth = {c: getattr(small_dataset, c) for c in radar_raman.COMPONENTS}
    pred = {c: v + 0.1 for c, v in truth.items()}
    stats = metrics.error_statistics(truth, pred)
    for c in radar_raman.COMPONENTS:
        assert np.isclose(stats[c]["max_error_mean"], 0.1)
        assert np.isclose(stats[c]["rmse_median"], 0.1)
        assert np.isclose(stats[c]["max_error_variance"], 0.0, atol=1e-12)
    ev = metrics.evaluate_components(truth, pred)
    assert set(ev["peaks"]) == {"rmse", "mae", "max_error", "r2"}
    per = metrics.per_spectrum_errors(truth, pred)
    assert per["noise"]["max_error"].shape == (len(small_dataset),)


def test_simple_api(tmp_path):
    test = radar_raman.generate(8, length=128, seed=0)
    assert isinstance(test, radar_raman.SyntheticDataset)
    m = radar_raman.load_model("standard")
    assert m.architecture == "standard" and radar_raman.load_model(m) is m
    res = radar_raman.evaluate("standard", test)
    assert set(res["peaks"]) == {"rmse", "mae", "max_error", "r2"}
    stats = radar_raman.evaluate(m, test.save(tmp_path / "t.h5"), statistics=True)
    assert "max_error_median" in stats["noise"]
    c = radar_raman.decompose(test.X, model=m, output=tmp_path / "c.db")
    np.testing.assert_allclose(radar_raman.load_components(tmp_path / "c.db").peaks, c.peaks)


@pytest.mark.slow
def test_simple_train(tmp_path):
    m = radar_raman.train(model="standard", init="pretrained", save_to=tmp_path / "m.keras", epochs=1,
                    steps_per_epoch=1, batch_size=4, length=64, learning_rate=1e-6,
                    validation_data=None, verbose=0)
    assert (tmp_path / "m.keras").exists() and m.history is not None


def test_cli_convert_and_formats(tmp_path, capsys):
    X = np.random.default_rng(0).random((2, 40))
    src = radar_raman.write(X, tmp_path / "a.csv", wavenumbers=np.linspace(100, 500, 40))
    cli.main(["convert", str(src), str(tmp_path / "a.spc")])
    np.testing.assert_allclose(radar_raman.read(tmp_path / "a.spc").intensities, X, rtol=1e-5)
    cli.main(["decompose", str(src), str(tmp_path / "a.spc"), "-o", str(tmp_path / "{stem}_out.db"),
              "-m", "standard"])
    assert (tmp_path / "a_out.db").exists()
    cli.main(["formats"])
    assert "jcamp" in capsys.readouterr().out
