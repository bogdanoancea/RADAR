# RADAR

**R**aman spectral **A**nalysis using **D**eep learning for **A**rtifact **R**emoval.

RADAR splits Raman spectra into four additive components with convolutional
neural networks:

```
spectrum = baseline + cosmic rays + noise + peaks
```

- **Baseline:** smooth background (fluorescence, etc.)
- **Cosmic rays:** narrow spikes
- **Noise:** random noise
- **Peaks:** the Raman bands, i.e. the corrected, denoised spectrum

The package contains:
- **Two pre-trained networks**, *standard* and *ensemble*, each trained on 240
  million synthetic spectra.
- **Training and evaluation tools:** a synthetic data generator for training
  and testing, training and fine-tuning routines, and evaluation metrics
  (RMSE, MAE, max error, R²).
- **File I/O for the common Raman formats:** SPC, JCAMP-DX, Renishaw WDF,
  Horiba LabSpec, TriVista, text exports, Excel, Parquet, SQLite, HDF5,
  MATLAB, JSON and NumPy.
- **Interfaces:** a command-line tool.

## Installation

```bash
cd RADAR
pip install .                 # core
pip install ".[formats]"      # + Excel / Parquet / Feather
pip install ".[renishaw]"     # + Renishaw .wdf
pip install -e ".[dev]"       # everything, for development
```

Requires Python ≥ 3.9 and TensorFlow ≥ 2.16 (Keras 3). Check the installation
with `radar-raman info`.

## Quick start

```python
import radar_raman
comps = radar_raman.decompose("spectra.spc")   # file (any supported format) or NumPy array
comps.baseline, comps.cosmic_rays, comps.noise, comps.peaks
comps.plot(0)
comps.save("spectra_components.h5")            # .csv .xlsx .parquet .db .mat .json .spc .jdx ...
```

## The API in eight functions

```python
radar_raman.decompose(spectra, model="ensemble", output=None)  # array / file -> Components
radar_raman.read(path)                                   # any format -> SpectralData
radar_raman.write(obj, path)                             # spectra, components or data sets
radar_raman.formats(show=True)                           # supported formats
radar_raman.load_model("ensemble")                       # pre-trained or saved model
radar_raman.train(data=None, model="ensemble", init=None, save_to="my.keras", epochs=..., ...)
radar_raman.generate(1000)                               # synthetic spectra with ground truth
radar_raman.evaluate(model, test_set)                    # RMSE, MAE, max error, R²
```

Saved models record the RADAR, Keras and TensorFlow versions they were made
with: `radar_raman.model_info("my.keras", show=True)` or `radar-raman info my.keras`.

## Supported file formats

| Read and write | Read only |
|---|---|
| CSV / TSV / TXT (incl. instrument exports, decimal commas), Excel, Parquet, Feather, SQLite, HDF5 / NetCDF-4, NumPy, MATLAB, JSON, Thermo Galactic **SPC**, **JCAMP-DX** | Renishaw **WDF**, Horiba **LabSpec** XML, Princeton **TriVista** TVF, RRUFF |

Instrument text exports from Renishaw WiRE, Horiba LabSpec, Ocean Insight,
WITec and RRUFF are recognised automatically. Raman maps keep their grid
shape (`comps.as_map("peaks")`, `comps.band_image(990, 1010)`).

## Command line

```bash
radar-raman decompose sample.wdf -o sample_components.h5
radar-raman decompose data/*.spc -o "results/{stem}.csv" -m standard
radar-raman convert measurement.spc measurement.jdx
radar-raman generate -n 1000 -o test.h5
radar-raman train -o my_model.keras --epochs 12 --steps 200000
radar-raman evaluate test.h5 -m my_model.keras --statistics
radar-raman formats
```

## Documentation

| | |
|---|---|
| **[User guide](docs/user_guide.md)** | everything in one place: installation, reading, decomposing, saving, models, training, fine-tuning, synthetic data, evaluation, CLI, troubleshooting |
| [Models](docs/models.md) | the two architectures and the pre-trained weights |
| [Synthetic data](docs/data_generation.md) | the data generator |
| [Command line](docs/cli.md) | all commands and options |
| [API reference](docs/api.md) | functions and classes |
| [Differences from the original code](docs/differences.md) | what changed and why |

To build the HTML documentation: `pip install ".[docs]" && mkdocs serve`.

## Examples

| Script | Shows |
|---|---|
| `examples/01_quickstart.py` | decompose, derived spectra, save, plot |
| `examples/02_decompose_file.py` | read a CSV, both models, save and reload, compare with ground truth |
| `examples/03_generate_data.py` | synthetic training and test data |
| `examples/04_train_new_model.py` | train from scratch, checkpoint, save, reload, evaluate (`--full` for the complete schedule) |
| `examples/05_finetune_pretrained.py` | fine-tune a pre-trained model |
| `examples/06_compare_models.py` | standard vs ensemble: RMSE, MAE, max error, R² |
| `examples/07_original_weights.py` | load weight files written by the original training scripts |
| `examples/08_file_formats.py` | formats, maps, band images, conversion, SQL queries |

## Tests

```bash
pip install -e ".[test,formats,renishaw]"
pytest                    # full suite
pytest -m "not slow"      # skip the training tests
```

The suite checks that the networks reproduce the outputs of the original
implementation (to within 1e-5), and that every writable format round-trips.
Optional tests on real instrument files run when `RADAR_SAMPLE_DATA` points to
the public test data of the SPC, JCAMP-DX and rosettasciio projects (see
`tests/test_formats.py`).

## Licence and credits

BSD 3-Clause, see [LICENSE](LICENSE). The networks, the synthetic data
generator and the pre-trained weights come from the original implementation
by **Joel Sjöberg**
([repository](https://github.com/JoelSjoberg/Comprehensive-Raman-Spectral-Analysis-Using-Deep-Learning)).
The Python package is by **Bogdan Oancea**. Both copyright notices must be
kept in redistributions. See [AUTHORS.md](AUTHORS.md).

Optional readers: `renishawWiRE` (MIT) and `rosettasciio` (GPL-3.0) are
separate packages that RADAR uses only if you install them.
