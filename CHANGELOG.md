# Changelog

## 0.1.0 (2026-10-24)

First release as a Python package, based on the original implementation by
J. Sjöberg.

**Models**
- `standard` and `ensemble` networks ported to Keras 3, with the pre-trained
  weights bundled and validated against the original implementation.
- `load_model` reads `.keras` files saved with a newer Keras version by
  rebuilding the network and loading only its weights.
- Saved models (`.keras`, `.weights.h5`) record the RADAR, Keras, TensorFlow,
  Python and NumPy versions, the architecture, hyper-parameters and training
  settings; `radar_raman.model_info()` / `radar-raman info MODEL` show them and flag
  version mismatches.

**API**
- Simple API: `radar_raman.decompose`, `read`, `write`, `formats`, `load_model`,
  `train`, `generate`, `evaluate`.
- `Decomposer` class: `decompose` (arrays, files, maps, spectra of different
  lengths), `train`, `evaluate`, `save`, `load`, `pretrained`.
- `Components`: derived spectra, maps (`as_map`, `band_image`), saving and
  plotting.

**File formats**
- Read and write: CSV / TSV / TXT (wide, rows and long layouts, instrument
  exports, decimal commas), Excel, Parquet, Feather, SQLite, HDF5 / NetCDF-4,
  NumPy, MATLAB, JSON, Thermo Galactic SPC (native) and JCAMP-DX (native,
  including compressed ASDF data).
- Read only: Renishaw WDF, Horiba LabSpec XML, TriVista TVF (optional
  packages), RRUFF.

**Data and metrics**
- Synthetic data generator with explicit random state; `SyntheticDataset`,
  train/test splitting, batch and `tf.data` generators. Fixes to the original
  generator (see `docs/differences.md`), with `legacy_scaling=True` to
  reproduce it exactly.
- Metrics: RMSE, MAE, max error and R², plus mean / median / variance
  summaries of the per-spectrum RMSE and max error.

**Tools and documentation**
- `radar-raman` command-line tool with the `decompose`, `convert`, `generate`,
  `train`, `evaluate`, `formats` and `info` commands.
- Tests, eight example scripts and a comprehensive user guide.
