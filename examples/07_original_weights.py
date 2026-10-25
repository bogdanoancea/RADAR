# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Use weight files produced by the original training scripts.

The original scripts (TrainStandard.py / TrainEnsemble.py) save Keras 2
weight files with ``model.save_weights("Models/....h5")``. RADAR loads them
directly; you only need to say which architecture they belong to.

Run:  python examples/07_original_weights.py [path/to/ensembleModelRes.h5]
"""
import sys
from pathlib import Path

import numpy as np

import radar
from radar._weights import pretrained_weights_path

# Default: the copy bundled with RADAR (identical to Models/ensembleModelRes.h5)
path = sys.argv[1] if len(sys.argv) > 1 else pretrained_weights_path("ensemble")

model = radar.load_model(path, architecture="ensemble")
print("loaded", path, "->", model)

# Convert to the Keras 3 formats
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
model.save(OUT / "ensemble_from_h5.keras")          # complete model
model.save(OUT / "ensemble_from_h5.weights.h5")     # weights only

x = radar.generate_dataset(2, length=1024, seed=0).X
same = np.allclose(model.decompose(x).peaks, radar.load_pretrained("ensemble").decompose(x).peaks)
print("same predictions as the bundled pre-trained model:", same)
