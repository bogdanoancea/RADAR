# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Regenerate tests/data/reference_outputs.npz from the ORIGINAL implementation.

Run from the root of the original repository (Scripts/, Models/), with
legacy Keras 2:  pip install tf-keras;  python make_reference.py
``reference_outputs_v1.npz`` is the previous fixture (from git history); its
X1024 and random-walk inputs are reused so the new file stays comparable.
"""
import os; os.environ["TF_USE_LEGACY_KERAS"]="1"
import numpy as np
from Scripts.essentials import make_standard, make_ensemble
from Scripts.generator import train_generator
old = np.load("reference_outputs_v1.npz")   # first version: X1024 and the random walks
np.random.seed(3)
X,_ = next(train_generator(length=1024,batch_size=6,min_width=10,max_width=300,increase_noise=True))
X = np.squeeze(np.array(X),-1)
assert np.array_equal(X, old["X1024"]), "X1024 not reproduced"
np.random.seed(11)
X2,_ = next(train_generator(length=1738,batch_size=4,min_width=10,max_width=300,increase_noise=True))
X2 = np.squeeze(np.array(X2),-1)
inputs = {"X1024": X, "X1738": X2, "W1738": old["X1738"]}   # W1738 = featureless random walks (old X1738)
out = dict(inputs)
for name,mk,w in [("standard",make_standard,"Models/standardModel.h5"),("ensemble",make_ensemble,"Models/ensembleModelRes.h5")]:
    m = mk(); m.load_weights(w)
    for k in inputs:
        p = m.predict(inputs[k][...,None], verbose=0)
        for c,arr in zip(["baseline","cosmic_rays","noise","peaks"],p): out[f"{name}_{k}_{c}"] = np.asarray(arr)
for k,v in out.items():
    if "_" in k: print(f"{k:32s} absmax={np.abs(v).max():.3g} nonzero_frac={np.count_nonzero(v)/v.size:.3f}")
np.savez_compressed("reference_outputs.npz", **out)
