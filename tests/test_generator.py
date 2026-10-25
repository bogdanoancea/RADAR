# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
import numpy as np
import pytest

import radar
from radar import generator as g


def test_components_ranges():
    rng = np.random.default_rng(0)
    for _ in range(20):
        bl, cr, nz, pk = g.generate_spectrum(800, rng=rng)
        assert bl.shape == cr.shape == nz.shape == pk.shape == (800,)
        assert cr.min() >= 0 and cr.max() < 1
        assert pk.min() >= 0 and pk.max() <= 1 + 1e-9
        assert nz[0] == 0 and nz[-1] == 0


def test_elementary_baselines_in_unit_interval():
    for b in g.make_baselines(500, rng=1):
        assert b.min() >= -1e-9 and b.max() <= 1 + 1e-9


@pytest.mark.parametrize("increase_noise", [False, True])
def test_sample_is_sum_of_targets(increase_noise):
    rng = np.random.default_rng(5)
    for _ in range(30):
        s, t = g.generate_sample(600, increase_noise=increase_noise, rng=rng)
        np.testing.assert_allclose(s, sum(t[c] for c in radar.COMPONENTS), atol=1e-10)


def test_test_setting_is_minmax_normalised():
    s, _ = g.generate_sample(600, increase_noise=True, rng=2)
    assert np.isclose(s.min(), 0) and np.isclose(s.max(), 1)


def test_legacy_scaling_reproduces_original_inconsistency():
    rng = np.random.default_rng(1)
    diffs = []
    for _ in range(20):
        s, t = g.generate_sample(600, rng=rng, legacy_scaling=True)
        diffs.append(np.abs(s - sum(t.values())).max())
    assert max(diffs) > 1e-3  # input is not the sum of the targets in the original


def test_short_spectra():
    s, t = g.generate_sample(64, rng=0)
    assert s.shape == (64,)


def test_dataset_reproducible_and_shapes():
    a = radar.generate_dataset(10, length=256, seed=7)
    b = radar.generate_dataset(10, length=256, seed=7)
    c = radar.generate_dataset(10, length=256, seed=8)
    assert a.X.shape == (10, 256) and len(a) == 10 and a.length == 256
    np.testing.assert_array_equal(a.X, b.X)
    assert not np.array_equal(a.X, c.X)
    X, y = a.as_keras()
    assert X.shape == (10, 256, 1) and len(y) == 4
    assert a.subset(slice(0, 3)).X.shape == (3, 256)


def test_train_test_independent():
    tr, te = radar.train_test_datasets(8, 4, length=200, seed=0)
    assert len(tr) == 8 and len(te) == 4
    assert tr.params["increase_noise"] is False and te.params["increase_noise"] is True


def test_batch_generator():
    gen = radar.batch_generator(length=128, batch_size=4, seed=0)
    X, y = next(gen)
    assert X.shape == (4, 128, 1) and X.dtype == np.float32
    assert len(y) == 4 and all(t.shape == (4, 128) for t in y)


def test_tf_dataset():
    ds = radar.make_tf_dataset(length=64, batch_size=2, seed=0)
    X, y = next(iter(ds))
    assert tuple(X.shape) == (2, 64, 1) and len(y) == 4
