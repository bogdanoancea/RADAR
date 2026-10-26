# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Read spectra from a file, decompose them with both models, save the results.

Run:  python examples/02_decompose_file.py

Command-line equivalent:
    radar-raman decompose examples/data/synthetic_spectra.csv -o components.h5 --model ensemble
"""
from pathlib import Path

import numpy as np

import radar_raman
HERE = Path(__file__).parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

# read_spectra handles .csv/.txt/.tsv (spectra in columns or rows, optional
# wavenumber axis and header), .npy/.npz and .h5 files.
data = radar_raman.read_spectra(HERE / "data" / "synthetic_spectra.csv")
print(data)            # SpectralData(n=5, length=1024, wavenumbers=400..1800, ...)
print("names:", data.names)

for name in ("standard", "ensemble"):
    model = radar_raman.load_pretrained(name)
    comps = model.decompose(data)          # wavenumbers are carried over
    # .npz / .h5 keep everything in one file; .csv writes one table per array
    comps.save(OUT / f"file_components_{name}.h5")
    comps.save(OUT / f"file_components_{name}.csv")
    print(f"{name:9s}: peaks range {comps.peaks.min():.1f} .. {comps.peaks.max():.1f} "
          f"(input units), residual {np.abs(comps.residual).max():.2g}")

# Reload a saved decomposition
c = radar_raman.load_components(OUT / "file_components_ensemble.h5")
print("reloaded:", c, "wavenumbers", c.wavenumbers[[0, -1]])

# Because the example spectra are synthetic, the true components are known.
# They are stored in the normalised scale, so compare in that scale:
truth = radar_raman.SyntheticDataset.load(HERE / "data" / "synthetic_spectra_truth.npz")
model = radar_raman.load_pretrained("ensemble")
pred = model.decompose(data, units="normalized")
print("errors vs. ground truth:")
for comp, m in radar_raman.evaluate_components(truth, pred).items():
    print(f"  {comp:12s} RMSE={m['rmse']:.4f}  MaxError={m['max_error']:.4f}  R2={m['r2']:.3f}")
