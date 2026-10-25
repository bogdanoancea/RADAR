# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Command-line interface: ``radar <command> ...``.

Commands
--------
``radar decompose INPUT [INPUT ...] -o OUTPUT [--model ensemble|standard|FILE]``
    Decompose spectra from any supported file and save the components
    (format from the output extension). With several inputs, ``-o`` is a
    folder or a pattern containing ``{stem}``.
``radar convert INPUT OUTPUT``
    Convert spectra (or a decomposition) between file formats.
``radar generate -n N -o OUTPUT``
    Write a synthetic data set with ground-truth components.
``radar train -o MODEL.keras``
    Train a new model, or fine-tune one with ``--init pretrained|FILE``.
``radar evaluate DATASET [--model ...] [--statistics]``
    RMSE, MAE, max error and R^2 per component on a synthetic data set.
``radar formats``
    List the supported file formats.
``radar info [MODEL]``
    Versions and bundled models; with MODEL, the software versions and
    settings the model was saved with.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional


def _quiet_tf():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _read_options(a) -> dict:
    opts = {}
    for k in ("format", "layout", "key", "sheet", "table", "query", "delimiter", "decimal"):
        v = getattr(a, k, None)
        if v is not None:
            opts[k] = v
    return opts


def _cmd_decompose(a):
    from .decomposer import load_model
    from .io import read, write

    model = load_model(a.model, architecture=a.architecture)
    inputs = [Path(p) for p in a.input]
    for inp in inputs:
        if len(inputs) == 1 and "{stem}" not in a.output and not Path(a.output).is_dir():
            out = Path(a.output)
        elif "{stem}" in a.output:
            out = Path(a.output.format(stem=inp.stem))
        else:
            out = Path(a.output) / f"{inp.stem}_components.h5"
        data = read(inp, **_read_options(a))
        comps = model.decompose(data, normalize=not a.no_normalize, units=a.units,
                                batch_size=a.batch_size)
        written = write(comps, out, **({"layout": a.out_layout} if a.out_layout else {}))
        shape = f", map {data.map_shape}" if data.map_shape else ""
        print(f"{inp.name}: {len(data)} spectra{shape} -> {written}")


def _cmd_convert(a):
    from .io import load_components, read, write

    opts = _read_options(a)
    try:
        obj = load_components(a.input, **{k: v for k, v in opts.items() if k == "format"})
        what = "decomposition"
    except Exception:
        obj = read(a.input, **opts)
        what = f"{len(obj)} spectra"
    out = write(obj, a.output, **({"layout": a.out_layout} if a.out_layout else {}))
    print(f"{a.input} ({what}) -> {out}")


def _cmd_generate(a):
    from .generator import generate_dataset

    kw = dict(min_width=5, max_width=256, max_num_peaks=5, increase_noise=False) if a.train else {}
    if a.min_width is not None:
        kw["min_width"] = a.min_width
    if a.max_width is not None:
        kw["max_width"] = a.max_width
    ds = generate_dataset(a.n, length=a.length, seed=a.seed, **kw)
    out = ds.save(a.output)
    print(f"Wrote {len(ds)} synthetic spectra of {ds.length} points -> {out}")


def _cmd_train(a):
    from .decomposer import train

    lr = "original" if a.learning_rate is None else a.learning_rate
    dec = train(
        a.data,
        model=a.architecture,
        init=a.init,
        save_to=a.output,
        epochs=a.epochs,
        steps_per_epoch=a.steps,
        batch_size=a.batch_size,
        length=a.length,
        learning_rate=lr,
        validation_size=a.validation_size,
        validation_data="synthetic" if a.validation_size > 0 else None,
        seed=a.seed,
        checkpoint_path=a.checkpoint,
        verbose=a.verbose,
    )
    print(f"Saved trained {dec.architecture} model -> {a.output}")


def _cmd_evaluate(a):
    from .decomposer import evaluate, load_model

    res = evaluate(load_model(a.model, architecture=a.architecture), a.dataset,
                   statistics=a.statistics)
    print(json.dumps(res, indent=2))


def _cmd_formats(a):
    from .io import formats

    formats(show=True)


def _cmd_info(a):
    import keras
    import numpy as np
    import tensorflow as tf

    from . import __version__
    from ._weights import PRETRAINED_FILES, pretrained_weights_path
    from .io._vendor import available

    if a.model:
        from .decomposer import model_info

        model_info(a.model, show=True)
        return

    print(f"radar {__version__}")
    print(f"tensorflow {tf.__version__}, keras {keras.__version__}, numpy {np.__version__}")
    for arch in PRETRAINED_FILES:
        p = pretrained_weights_path(arch)
        print(f"pre-trained '{arch}': {p} ({'found' if p.exists() else 'MISSING'})")
    opt = available()
    print("optional vendor readers:", ", ".join(opt) if opt else "none installed")


def _add_read_options(p):
    p.add_argument("--format", default=None, help="input format (default: from the extension)")
    p.add_argument("--layout", default=None, choices=["auto", "wide", "columns", "rows", "long"],
                   help="table layout of text/Excel/Parquet/SQLite input")
    p.add_argument("--key", default=None, help="array name in .npz/.h5/.mat/.json input")
    p.add_argument("--sheet", default=None, help="Excel sheet name or index")
    p.add_argument("--table", default=None, help="SQLite table")
    p.add_argument("--query", default=None, help="SQLite query returning the spectra")
    p.add_argument("--delimiter", default=None, help="text field separator (default: detected)")
    p.add_argument("--decimal", default=None, choices=[".", ","], help="text decimal mark")
    p.add_argument("--out-layout", default=None, choices=["wide", "long"],
                   help="table layout of text/Parquet/SQLite output")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="radar", description="RADAR: deep-learning decomposition of Raman spectra")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("decompose", help="decompose spectra from files")
    d.add_argument("input", nargs="+")
    d.add_argument("-o", "--output", required=True,
                   help="output file; for several inputs a folder or a pattern with {stem}")
    d.add_argument("-m", "--model", default="ensemble", help="'ensemble', 'standard' or a model file")
    d.add_argument("--architecture", default=None, help="required for weight-only files")
    d.add_argument("--units", default="original", choices=["original", "normalized"])
    d.add_argument("--no-normalize", action="store_true", help="input is already in [0, 1]")
    d.add_argument("--batch-size", type=int, default=32)
    _add_read_options(d)
    d.set_defaults(func=_cmd_decompose)

    c = sub.add_parser("convert", help="convert spectra or a decomposition between formats")
    c.add_argument("input")
    c.add_argument("output")
    _add_read_options(c)
    c.set_defaults(func=_cmd_convert)

    g = sub.add_parser("generate", help="generate synthetic spectra with known components")
    g.add_argument("-n", type=int, required=True)
    g.add_argument("-o", "--output", required=True)
    g.add_argument("--length", type=int, default=1024)
    g.add_argument("--min-width", type=int, default=None)
    g.add_argument("--max-width", type=int, default=None)
    g.add_argument("--train", action="store_true", help="training settings instead of test settings")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=_cmd_generate)

    t = sub.add_parser("train", help="train a new model or fine-tune an existing one")
    t.add_argument("-o", "--output", required=True, help="output .keras file")
    t.add_argument("-a", "--architecture", default="ensemble", choices=["ensemble", "standard"])
    t.add_argument("--init", default=None, help="'pretrained' or a model file to fine-tune")
    t.add_argument("--data", default=None, help="fixed training set (from 'radar generate')")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--steps", type=int, default=1000, help="steps per epoch (synthetic stream)")
    t.add_argument("--batch-size", type=int, default=100)
    t.add_argument("--length", type=int, default=750)
    t.add_argument("--learning-rate", type=float, default=None,
                   help="constant learning rate (default: original schedule)")
    t.add_argument("--validation-size", type=int, default=1000)
    t.add_argument("--checkpoint", default=None, help="save best model to this .keras file")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--verbose", type=int, default=1)
    t.set_defaults(func=_cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a model on a synthetic data set")
    e.add_argument("dataset")
    e.add_argument("-m", "--model", default="ensemble")
    e.add_argument("--architecture", default=None)
    e.add_argument("--statistics", action="store_true",
                   help="mean/median/variance of per-spectrum RMSE and max error")
    e.set_defaults(func=_cmd_evaluate)

    f = sub.add_parser("formats", help="list supported file formats")
    f.set_defaults(func=_cmd_formats)

    i = sub.add_parser("info", help="installation info, or the provenance of a saved model")
    i.add_argument("model", nargs="?", default=None,
                   help="model file (or 'ensemble'/'standard'): show the software versions it was saved with")
    i.set_defaults(func=_cmd_info)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    _quiet_tf()
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
