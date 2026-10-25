# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Generate synthetic training and test data with known components.

Run:  python examples/03_generate_data.py

Command-line equivalent:
    radar generate -n 1000 --length 1024 -o test_set.npz          # test settings
    radar generate -n 1000 --length 750 --train -o train_set.npz  # training settings
"""
from pathlib import Path

import numpy as np

import radar
from radar import generator

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# --- Original test set (GenerateTestData.ipynb): length 1024, peak
#     widths 10-300 points, increased noise, spectra min-max normalised.
test = radar.generate_dataset(200, length=1024, min_width=10, max_width=300,
                              increase_noise=True, seed=0)
print("test set:", test.X.shape, "params:", test.params)
print("input == sum of components:",
      np.allclose(test.X, test.baseline + test.cosmic_rays + test.noise + test.peaks, atol=1e-6))
test.save(OUT / "test_set.npz")

# --- Independent train/test sets with the original settings
train, test2 = radar.train_test_datasets(n_train=500, n_test=100, length=750, seed=1)
print("train:", train.X.shape, "test:", test2.X.shape)

# --- Arrays for Keras: X has shape (n, length, 1), y = [baseline, cosmic_rays, noise, peaks]
X, y = train.as_keras()
print("keras arrays:", X.shape, [t.shape for t in y])

# --- Infinite stream of training batches (used by Decomposer.train)
batches = radar.batch_generator(length=750, batch_size=100, seed=2)
Xb, yb = next(batches)
print("one batch:", Xb.shape, [t.shape for t in yb])

# --- Individual components
rng = np.random.default_rng(3)
baseline = generator.generate_baseline(1000, rng=rng)
peaks = generator.generate_peaks(1000, max_num_peaks=6, min_peak_width=10, max_peak_width=120, rng=rng)
rays = generator.generate_cosmic_rays(1000, max_num=10, rng=rng)
noise = generator.generate_noise(1000, rng=rng)
print("components:", [a.shape for a in (baseline, peaks, rays, noise)])

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(5, 1, figsize=(8, 8), sharex=True)
    i = 0
    for a, (lab, v) in zip(ax, [("spectrum", test.X[i]), ("baseline", test.baseline[i]),
                                ("cosmic rays", test.cosmic_rays[i]), ("noise", test.noise[i]),
                                ("peaks", test.peaks[i])]):
        a.plot(v, lw=0.8)
        a.set_ylabel(lab)
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_example.png", dpi=120)
except ImportError:
    pass
