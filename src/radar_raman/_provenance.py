# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Provenance of saved models: software versions, architecture, training.

Every model saved by RADAR carries a JSON record (``radar_info``):

* ``.keras`` files: an extra ``radar_info.json`` entry inside the archive
  (Keras ignores it when loading);
* ``.weights.h5`` files: a ``radar_info`` attribute on the HDF5 root.

The record holds the RADAR, Keras, TensorFlow, Python, NumPy and h5py
versions used to save the model, the platform, the architecture and its
hyper-parameters, the number of parameters, the save date and, if the model
was trained with RADAR, the training settings. :func:`read_info` returns it
(also for files without a record: ``.keras`` files from plain Keras and the
Keras 2 weight files of the original scripts), :func:`compatibility` compares
it with the installed versions.
"""

from __future__ import annotations

import json
import platform
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

INFO_NAME = "radar_info.json"
INFO_FORMAT = 1


def environment() -> Dict[str, str]:
    """Versions of the software RADAR runs on."""
    import h5py
    import keras
    import numpy as np

    from ._version import __version__

    try:
        import tensorflow as tf

        tf_version = tf.__version__
    except Exception:  # pragma: no cover
        tf_version = "not installed"
    return {
        "radar": __version__,
        "keras": keras.__version__,
        "keras_backend": keras.backend.backend(),
        "tensorflow": tf_version,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
    }


def hyperparameters(model) -> Dict[str, float]:
    """Hyper-parameters of a RADAR network, read from its layers."""
    try:
        enc = model.get_layer("enc_conv1")
        key = model.get_layer("key_conv")
    except ValueError:
        return {}
    reg = getattr(enc, "kernel_regularizer", None)
    cfg = reg.get_config() if reg is not None else {}
    return {
        "scaler": int(enc.filters),
        "dim_red_size": int(key.filters),
        "l1_param": float(cfg.get("l1", 1e-6)),
        "l2_param": float(cfg.get("l2", 1e-6)),
    }


def make_info(decomposer) -> dict:
    """The provenance record of a :class:`~radar_raman.Decomposer`."""
    info = {
        "radar_info_format": INFO_FORMAT,
        "saved_with": environment(),
        "date_saved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "architecture": decomposer.architecture,
        "hyperparameters": hyperparameters(decomposer.model),
        "n_parameters": int(decomposer.model.count_params()),
        "origin": getattr(decomposer, "origin", ""),
    }
    training = getattr(decomposer, "training_info", None)
    if training:
        info["training"] = training
    return info


def write_info(path, info: dict) -> None:
    """Attach ``info`` to a saved ``.keras`` or ``.weights.h5`` file."""
    path = Path(path)
    text = json.dumps(info, indent=2, default=str)
    if path.suffix == ".keras":
        with zipfile.ZipFile(path, "a") as z:
            if INFO_NAME not in z.namelist():
                z.writestr(INFO_NAME, text)
    elif path.name.endswith(".h5"):
        import h5py

        with h5py.File(path, "a") as f:
            f.attrs["radar_info"] = text


def read_info(path) -> dict:
    """Provenance record of a model file (empty dict if nothing is known).

    For files saved without a RADAR record the available information is
    returned: the Keras version from a ``.keras`` archive's
    ``metadata.json``, or the Keras version and backend of a Keras 2 ``.h5``
    weight file.
    """
    path = Path(path)
    if path.suffix == ".keras":
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if INFO_NAME in names:
                return json.loads(z.read(INFO_NAME))
            info = {"saved_with": {}}
            if "metadata.json" in names:
                meta = json.loads(z.read("metadata.json"))
                info["saved_with"]["keras"] = meta.get("keras_version", "")
                info["date_saved"] = meta.get("date_saved", "")
            if "config.json" in names:
                cfg = json.loads(z.read("config.json")).get("config", {})
                info["model_name"] = cfg.get("name", "")
            info["note"] = "saved without RADAR provenance (only the Keras version is known)"
            return info
    if path.suffix in (".h5", ".hdf5"):
        import h5py

        with h5py.File(path, "r") as f:
            if "radar_info" in f.attrs:
                raw = f.attrs["radar_info"]
                return json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            if "keras_version" in f.attrs:
                kv = f.attrs["keras_version"]
                be = f.attrs.get("backend", b"")
                kv = kv.decode() if isinstance(kv, bytes) else str(kv)
                be = be.decode() if isinstance(be, bytes) else str(be)
                return {"saved_with": {"keras": kv, "keras_backend": be},
                        "note": "Keras 2 weight file (e.g. from the original training scripts); "
                                "load with radar_raman.load_model(path, architecture=...)"}
    return {}


def _vtuple(v: str):
    out = []
    for part in str(v).split("+")[0].split(".")[:3]:
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out + [0] * (3 - len(out)))


def compatibility(info: dict, installed: Optional[Dict[str, str]] = None) -> List[str]:
    """Human-readable notes comparing a saved model with the installed software.

    An empty list means no differences that matter. A model saved with a
    *newer* Keras or TensorFlow than the installed one is flagged; older
    versions load normally.
    """
    installed = installed or environment()
    saved = info.get("saved_with", {})
    notes = []
    for pkg in ("keras", "tensorflow", "radar"):
        s, i = saved.get(pkg), installed.get(pkg)
        if not s or not i or i == "not installed":
            continue
        if _vtuple(s)[:2] > _vtuple(i)[:2]:
            notes.append(f"saved with {pkg} {s}, installed {i}: upgrade with `pip install -U "
                         f"\"{pkg if pkg != 'radar' else 'radar-raman'}>={s}\"` to load it natively")
        elif pkg == "keras" and _vtuple(s)[0] < 3:
            notes.append(f"saved with Keras {s} (Keras 2): only the weights can be used; "
                         "load with radar_raman.load_model(path, architecture=...)")
    return notes


def describe(info: dict) -> str:
    """Multi-line text summary of a provenance record."""
    if not info:
        return "no provenance information"
    lines = []
    sw = info.get("saved_with", {})
    if info.get("architecture"):
        hp = info.get("hyperparameters") or {}
        hp_txt = ", ".join(f"{k}={v}" for k, v in hp.items())
        lines.append(f"architecture : {info['architecture']}" + (f" ({hp_txt})" if hp_txt else ""))
    if info.get("n_parameters"):
        lines.append(f"parameters   : {info['n_parameters']:,}")
    if info.get("origin"):
        lines.append(f"origin       : {info['origin']}")
    if info.get("date_saved"):
        lines.append(f"saved        : {info['date_saved']}")
    if sw:
        lines.append("saved with   : " + ", ".join(f"{k} {v}" for k, v in sw.items() if v))
    rw = info.get("running_with", {})
    if rw:
        lines.append("running with : " + ", ".join(f"{k} {v}" for k, v in rw.items() if v))
    tr = info.get("training")
    if tr:
        lines.append("training     : " + ", ".join(f"{k}={v}" for k, v in tr.items()))
    if info.get("note"):
        lines.append(f"note         : {info['note']}")
    return "\n".join(lines)
