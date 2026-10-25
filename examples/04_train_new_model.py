# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Train a new RADAR model on synthetic spectra, save it, reload it, evaluate it.

Run:  python examples/04_train_new_model.py [--full]

Without --full this runs a SHORT demonstration (a few minutes on a CPU) that
does not produce a useful model. The bundled pre-trained models were trained
for 12 epochs x 200 000 steps of 100 spectra = 240 million spectra, which
needs a GPU and days of compute:

    model.train(epochs=12, steps_per_epoch=200_000, batch_size=100, length=750)

Command-line equivalent:
    radar train --architecture ensemble --epochs 12 --steps 200000 -o my_model.keras
"""
import sys
from pathlib import Path

import radar

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
FULL = "--full" in sys.argv

# 1. A new, randomly initialised network ("ensemble" or "standard").
#    Hyper-parameters can be changed, e.g. radar.Decomposer.new("ensemble", scaler=16).
model = radar.Decomposer.new("ensemble")
print(model)

# 2. Train on an endless stream of freshly generated synthetic spectra.
#    learning_rate="original" stretches the original 12-step decay over the run.
settings = dict(epochs=12, steps_per_epoch=200_000) if FULL else dict(epochs=2, steps_per_epoch=20)
history = model.train(
    **settings,
    batch_size=100,
    length=750,                    # original training spectrum length
    min_width=5, max_width=256,    # band widths (points)
    max_num_peaks=5,
    validation_size=200 if not FULL else 1000,
    checkpoint_path=str(OUT / "best_model.keras"),   # best model by validation loss
    verbose=2,
)
print("final training loss:", history.history["loss"][-1])

# 3. Save / reload. '.keras' keeps architecture, weights and optimiser state.
path = model.save(OUT / "my_model.keras")
model2 = radar.load_model(path)
print("reloaded:", model2)

# 4. Evaluate on an independent synthetic test set and compare with the
#    pre-trained model.
test = radar.generate_dataset(200, length=1024, seed=99)
for name, m in [("new", model2), ("pre-trained", radar.load_pretrained("ensemble"))]:
    res = m.evaluate(test)
    print(name, {k: round(v["r2"], 3) for k, v in res.items()})
