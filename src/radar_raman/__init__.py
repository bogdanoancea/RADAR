# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""RADAR -- Raman spectral Analysis using Deep learning for Artifact Removal.

Decompose Raman spectra into four additive components with convolutional
neural networks::

    spectrum = baseline + cosmic_rays + noise + peaks

The whole API in eight functions::

    import radar_raman
    comps = radar_raman.decompose("map.wdf")            # file or array -> Components
    comps = radar_raman.decompose(X, model="standard", output="result.h5")

    data  = radar_raman.read("spectra.spc")             # any supported format
    radar_raman.write(comps, "result.parquet")          # spectra / components / data sets

    model = radar_raman.load_model("ensemble")          # pre-trained, or a saved file
    model = radar_raman.train(epochs=12, save_to="my.keras")  # new or fine-tuned network

    test  = radar_raman.generate(1000)                  # synthetic spectra with ground truth
    radar_raman.evaluate(model, test)                   # RMSE, MAE, max error, R^2

``radar_raman.formats(show=True)`` lists the supported file formats;
``radar_raman.model_info("my.keras", show=True)`` shows which software versions a
saved model was made with.

The networks, the synthetic training data generator and the pre-trained
weights come from the original implementation by Joel Sjöberg.
"""

from ._version import __version__
from .layers import CustomPad, EdgePad1D, ExpandLastDim, ResidualSubtract  # registers layers
from . import losses  # registers losses
from .models import (
    ARCHITECTURES,
    COMPONENTS,
    build_ensemble_model,
    build_model,
    build_standard_model,
    compile_model,
)
from .components import Components
from .decomposer import (
    Decomposer,
    decompose,
    evaluate,
    load_model,
    load_pretrained,
    model_info,
    new_model,
    train,
)
from .generator import (
    SyntheticDataset,
    batch_generator,
    generate_dataset,
    generate_sample,
    make_tf_dataset,
    train_test_datasets,
)
from .io import (
    SpectralData,
    formats,
    load_components,
    load_dataset,
    read,
    read_spectra,
    save_components,
    save_dataset,
    save_spectra,
    write,
)
from . import metrics
from .metrics import error_statistics, evaluate_components

#: Synthetic spectra with known components (alias of :func:`generate_dataset`).
generate = generate_dataset

#: DOI of this software release (Zenodo).
__doi__ = "10.5281/zenodo.23760379"

#: Reference for the software.
SOFTWARE_CITATION = (
    "J. Sjöberg, B. Oancea, RADAR: Raman spectral Analysis using Deep learning "
    f"for Artifact Removal (Python package), version {__version__}, Zenodo (2026). "
    "https://doi.org/10.5281/zenodo.23760379"
)

#: Reference for the method.
METHOD_CITATION = (
    "J. Sjöberg, N. Siminea, A. Păun, A. Lita, M. Larion, I. Petre, "
    '"RADAR: Raman Spectral Analysis Using Deep Learning for Artifact Removal", '
    "Advanced Optical Materials (2025) 2500736. https://doi.org/10.1002/adom.202500736"
)

CITATION = SOFTWARE_CITATION + "\n" + METHOD_CITATION


def citation() -> str:
    """Return the references to cite when using RADAR: the software and the method."""
    return CITATION


__all__ = [
    "__version__",
    "__doi__",
    # the simple API
    "decompose",
    "read",
    "write",
    "formats",
    "load_model",
    "train",
    "generate",
    "evaluate",
    # classes
    "Decomposer",
    "Components",
    "SpectralData",
    "SyntheticDataset",
    # more
    "load_pretrained",
    "model_info",
    "new_model",
    "generate_dataset",
    "generate_sample",
    "train_test_datasets",
    "batch_generator",
    "make_tf_dataset",
    "read_spectra",
    "save_spectra",
    "save_components",
    "load_components",
    "save_dataset",
    "load_dataset",
    "evaluate_components",
    "error_statistics",
    "metrics",
    "ARCHITECTURES",
    "COMPONENTS",
    "build_model",
    "build_standard_model",
    "build_ensemble_model",
    "compile_model",
    "EdgePad1D",
    "CustomPad",
    "ExpandLastDim",
    "ResidualSubtract",
    "citation",
    "CITATION",
]
