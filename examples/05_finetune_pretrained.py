# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Fine-tune a pre-trained model on a fixed (e.g. instrument-specific) data set.

Run:  python examples/05_finetune_pretrained.py

Fine-tuning needs spectra whose true components are known. Here they are
synthetic, generated with broader bands than the original training data --
e.g. to adapt the model to a spectrometer that samples bands more densely.
Any (X, [baseline, cosmic_rays, noise, peaks]) arrays work.

The sizes are kept small so the script finishes in a few minutes on a CPU;
for a real fine-tuning use thousands of spectra and several epochs (GPU).
"""
from pathlib import Path

import radar

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# Data resembling the target instrument: broader bands
train, test = radar.train_test_datasets(
    n_train=160, n_test=64, length=1024, seed=0,
    train_kwargs=dict(min_width=20, max_width=400),
    test_kwargs=dict(min_width=20, max_width=400),
)

model = radar.load_pretrained("ensemble")
before = model.evaluate(test)

# Use a small constant learning rate for fine-tuning (the original schedule
# starts at 1e-3, which would destroy the pre-trained weights).
model.train(train, epochs=1, batch_size=16, learning_rate=1e-5,
            validation_data=test, verbose=2)
after = model.evaluate(test)

for comp in radar.COMPONENTS:
    print(f"{comp:12s} R2 before {before[comp]['r2']:.3f}  after {after[comp]['r2']:.3f}")

# NOTE: with this toy amount of data the scores can go slightly DOWN (batch-
# normalisation statistics are re-estimated from a few small batches). Always
# compare before/after on held-out data, as here, and keep the pre-trained
# model if fine-tuning does not help.

model.save(OUT / "ensemble_finetuned.keras")
