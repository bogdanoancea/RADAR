# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pathlib import Path

import numpy as np
import pytest

import radar_raman
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def reference():
    """Outputs of the ORIGINAL implementation (Keras 2 / tf_keras) on fixed inputs.

    Generated once with the original scripts (Scripts/essentials.py,
    make_standard / make_ensemble + the original .h5 weights); see
    tests/data/README.md.
    """
    with np.load(DATA_DIR / "reference_outputs.npz") as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="session")
def pretrained():
    return {a: radar_raman.load_pretrained(a) for a in ("standard", "ensemble")}


@pytest.fixture(scope="session")
def small_dataset():
    return radar_raman.generate_dataset(16, length=512, seed=123)
