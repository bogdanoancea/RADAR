# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Quick start: decompose spectra with a pre-trained RADAR model.

Run:  python examples/01_quickstart.py
"""
from pathlib import Path

import numpy as np

import radar_raman
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# 1. Some spectra: an array of shape (n_spectra, length) or (length,).
#    Here: synthetic test spectra, scaled to arbitrary intensity units.
data = radar_raman.generate_dataset(4, length=1024, seed=0)
spectra = 2500 * data.X + 300

# 2. Load a model: the pre-trained 'ensemble' (or 'standard') network.
model = radar_raman.load_pretrained("ensemble")
print(model)

# 3. Decompose. Spectra are min-max normalised internally and the components
#    are returned in the input units, so that they add up to the input.
comps = model.decompose(spectra)
print(comps)
print("baseline   :", comps.baseline.shape)
print("cosmic rays:", comps.cosmic_rays.shape)
print("noise      :", comps.noise.shape)
print("peaks      :", comps.peaks.shape)
print("max |input - sum of components|:", np.abs(comps.residual).max())

# Derived spectra
clean = comps.corrected            # = peaks: baseline, spikes and noise removed
denoised = comps.denoised          # input - noise
despiked = comps.despiked          # input - cosmic rays

# 4. Save the result (.npz, .h5 or .csv) and plot one spectrum.
comps.save(OUT / "quickstart_components.npz")
try:
    import matplotlib

    matplotlib.use("Agg")
    fig = comps.plot(index=0)
    fig.savefig(OUT / "quickstart_components.png", dpi=120)
    print("figure saved to", OUT / "quickstart_components.png")
except ImportError:
    print("install matplotlib to plot: pip install matplotlib")

# One-liner alternative (the pre-trained model is cached between calls):
comps2 = radar_raman.decompose(spectra[0], model="ensemble")
print("single spectrum ->", comps2)
