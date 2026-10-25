# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Compare the standard and ensemble models on synthetic test data.

Run:  python examples/06_compare_models.py
"""
from pathlib import Path

import radar
from radar.plotting import plot_comparison

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

test = radar.generate_dataset(500, length=1024, seed=2024)   # original test settings

print(f"{'component':12s} {'model':9s} {'RMSE':>8s} {'MAE':>8s} {'MaxErr':>8s} {'R2':>7s}")
preds = {}
for name in ("standard", "ensemble"):
    model = radar.load_pretrained(name)
    preds[name] = model.decompose(test.X, normalize=False)   # test data are already in [0, 1]
    for comp, m in radar.evaluate_components(test, preds[name]).items():
        print(f"{comp:12s} {name:9s} {m['rmse']:8.4f} {m['mae']:8.4f} {m['max_error']:8.4f} {m['r2']:7.3f}")

# Mean / median / variance over spectra of the per-spectrum RMSE and max error
print()
print(f"{'component':12s} {'model':9s} {'RMSE mean':>10s} {'median':>8s} {'var':>9s} "
      f"{'MaxErr mean':>12s} {'median':>8s} {'var':>9s}")
for name, comps in preds.items():
    for comp, s in radar.metrics.error_statistics(test, comps).items():
        print(f"{comp:12s} {name:9s} {s['rmse_mean']:10.4f} {s['rmse_median']:8.4f} {s['rmse_variance']:9.2e} "
              f"{s['max_error_mean']:12.4f} {s['max_error_median']:8.4f} {s['max_error_variance']:9.2e}")

try:
    import matplotlib

    matplotlib.use("Agg")
    for name, comps in preds.items():
        plot_comparison(test, comps, index=0).savefig(OUT / f"compare_{name}.png", dpi=120)
    print("figures saved to", OUT)
except ImportError:
    pass
