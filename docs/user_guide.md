# RADAR user guide

RADAR (**R**aman spectral **A**nalysis using **D**eep learning for
**A**rtifact **R**emoval) splits Raman spectra into four additive components
with convolutional neural networks:

```
spectrum = baseline + cosmic rays + noise + peaks
```

This guide covers everything the package can do, from a one-line
decomposition to training your own network. Sections can be read
independently.

**Contents**

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [The API at a glance](#4-the-api-at-a-glance)
5. [Reading spectra](#5-reading-spectra)
6. [Decomposing spectra](#6-decomposing-spectra)
7. [Working with the results](#7-working-with-the-results)
8. [Saving results and converting files](#8-saving-results-and-converting-files)
9. [Models](#9-models)
10. [Training a new model](#10-training-a-new-model)
11. [Fine-tuning a model](#11-fine-tuning-a-model)
12. [Synthetic data](#12-synthetic-data)
13. [Evaluating a model](#13-evaluating-a-model)
14. [Command line](#14-command-line)
15. [Practical advice and limitations](#15-practical-advice-and-limitations)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Overview

| Component | Attribute | Typical origin | What the network looks for |
|---|---|---|---|
| Baseline | `baseline` | fluorescence, substrate, optics | broad, smooth background |
| Cosmic rays | `cosmic_rays` | high-energy particles hitting the detector | spikes 1–2 points wide |
| Noise | `noise` | shot and read-out noise | point-to-point random fluctuation |
| Peaks | `peaks` | Raman bands | the corrected, denoised spectrum |

Two pre-trained networks are included:

| Model | Parameters | How it works | Components sum to the input? |
|---|---|---|---|
| `ensemble` (default) | 280 592 | Two stages. Stage 1 estimates the baseline and cosmic rays and removes them; stage 2 separates peaks and noise. | approximately |
| `standard` | 585 299 | One stage with heads for baseline, cosmic rays and peaks. The noise is defined as what remains. | exactly |

Both were trained on 240 million synthetic spectra with known components:
12 epochs × 200 000 batches of 100 spectra of 750 points. They work on
spectra of **any length** and from **any instrument**, because every spectrum
is scaled to [0, 1] before it goes through the network.

## 2. Installation

```bash
pip install /path/to/RADAR                    # core package
pip install "/path/to/RADAR[formats]"         # + Excel, Parquet, Feather
pip install "/path/to/RADAR[renishaw]"        # + Renishaw .wdf files
pip install "/path/to/RADAR[plot]"            # + plotting (matplotlib)
pip install -e "/path/to/RADAR[dev]"          # everything, editable, for development
```

| Extra | Installs | Needed for |
|---|---|---|
| (none) | numpy, scipy, h5py, tensorflow ≥ 2.16, keras ≥ 3 | everything else |
| `formats` | pandas, pyarrow, openpyxl | `.xlsx`, `.parquet`, `.feather` |
| `renishaw` | renishawWiRE (MIT) | `.wdf` |
| `vendor-gpl` | rosettasciio (**GPL-3.0**) | Horiba LabSpec `.xml`, TriVista `.tvf` |
| `plot` | matplotlib | `Components.plot`, `radar.plotting` |
| `test` / `docs` | pytest / mkdocs | running the tests, building these pages |

`rosettasciio` is GPL-licensed. RADAR only imports it at run time when such a
file is read, and does not depend on it or include it.

**GPU.** Decomposition runs well on a CPU: about 30–50 spectra of 1024 points
per second on two cores. Training from scratch needs a GPU. On Linux use
`pip install "tensorflow[and-cuda]"`, on Apple silicon
`pip install tensorflow-metal`.

**Check the installation:**

```bash
radar info       # versions, location of the pre-trained weights, optional readers
radar formats    # supported file formats
```

## 3. Quick start

```python
import radar

comps = radar.decompose("my_spectra.spc")      # any supported file, or a NumPy array
comps.peaks                                    # corrected spectra (same units as the input)
comps.plot(0)                                  # input and components of spectrum 0
comps.save("my_spectra_components.h5")         # or .csv, .xlsx, .parquet, .db, ...
```

In one line, with the result saved:

```python
radar.decompose("map.wdf", output="map_components.h5")
```

## 4. The API at a glance

Most tasks need only these eight functions:

| Function | Purpose |
|---|---|
| `radar.decompose(spectra, model="ensemble", output=None, **options)` | decompose an array, a `SpectralData` or a file; optionally save the result |
| `radar.read(path, format=None, **options)` | read spectra from any supported file → `SpectralData` |
| `radar.write(obj, path, **options)` | write spectra, a decomposition or a data set; format from the extension |
| `radar.formats(show=False)` | list the supported formats |
| `radar.load_model(model="ensemble")` | a pre-trained model by name, or a saved model from a file → `Decomposer` |
| `radar.train(data=None, model="ensemble", init=None, save_to=None, **options)` | train a new network or fine-tune one → `Decomposer` |
| `radar.generate(n, length=1024, **options)` | synthetic spectra with known components → `SyntheticDataset` |
| `radar.evaluate(model, dataset, statistics=False)` | RMSE, MAE, max error and R² per component |

`radar.model_info(path)` tells which software versions and settings a saved
model was made with (§9.2).

The four classes behind them:

| Class | What it is |
|---|---|
| `SpectralData` | spectra read from a file: intensities, axis, names, positions, map shape, metadata |
| `Decomposer` | a network plus pre/post-processing, training, evaluation and saving |
| `Components` | the result of a decomposition |
| `SyntheticDataset` | synthetic spectra together with their true components |

Everything else (`radar.models`, `radar.generator`, `radar.io`,
`radar.metrics`, `radar.training`, `radar.plotting`) is available for
finer control; see the [API reference](api.md).

## 5. Reading spectra

```python
data = radar.read("file.ext")      # the format follows the extension
data = radar.read("file.ext", format="csv", layout="rows")   # or set it explicitly
```

### 5.1 Supported formats

| Format | Extensions | Read | Write | Notes |
|---|---|---|---|---|
| Delimited text | `.csv` `.tsv` `.tab` `.txt` `.dat` `.asc` `.prn` `.xy` | ✓ | ✓ | the delimiter (`,` `;` tab `\|` or blanks) and decimal comma are detected; instrument exports are recognised (§5.3) |
| RRUFF | `.rruff` (and `.txt`) | ✓ | | `##` header lines, *x, y* pairs |
| Excel | `.xlsx` `.xlsm` `.xls` | ✓ | ✓ | needs `formats` extra; `sheet=` index, name or `"all"` |
| Parquet | `.parquet` `.pq` | ✓ | ✓ | needs `formats` extra; long (tidy) table by default |
| Feather / Arrow | `.feather` `.arrow` | ✓ | ✓ | needs `formats` extra |
| SQLite | `.db` `.sqlite` `.sqlite3` | ✓ | ✓ | table `spectra` + `radar_metadata`; `table=` or any SQL `query=` |
| HDF5 / NetCDF-4 | `.h5` `.hdf5` `.he5` `.nc` `.nc4` | ✓ | ✓ | `key=` names the data set; maps (3-D arrays) are recognised |
| NumPy | `.npy` `.npz` | ✓ | ✓ | `key=` for `.npz` |
| MATLAB | `.mat` | ✓ | ✓ | v5–v7.3; variables or (nested) structs; `key=` |
| JSON | `.json` | ✓ | ✓ | RADAR layout, `{"x": [...], "y": [...]}` or a list of such objects |
| Thermo Galactic GRAMS SPC | `.spc` | ✓ | ✓ | new (0x4B) and old (0x4D) format; single, multi-file and XY files |
| JCAMP-DX | `.jdx` `.dx` `.jcamp` `.jcm` | ✓ | ✓ | 4.24 / 5.x; AFFN and compressed ASDF data; compound files |
| Renishaw WiRE | `.wdf` | ✓ | | needs `renishaw` extra; single spectra, series, line scans, maps |
| Horiba LabSpec 6 | `.xml` | ✓ | | needs `vendor-gpl` extra; spectra, line scans, maps |
| Princeton Instruments TriVista | `.tvf` | ✓ | | needs `vendor-gpl` extra |

The SPC, JCAMP-DX and vendor readers were checked against real instrument
files: 27 SPC files, the JCAMP-DX test collections (Raman spectra and the
compressed-format tests), 14 Renishaw `.wdf`, 9 LabSpec `.xml` and 9 TriVista
`.tvf` files. Instrument software that cannot save any of these formats can
almost always export SPC, JCAMP-DX or text.

**Formats that are not read directly:** WITec `.wip` projects, Bruker OPUS
binaries, Thermo OMNIC `.spa`, Andor `.sif` and Princeton `.spe` camera
files. Export them from the instrument software as SPC, JCAMP-DX, MATLAB or
text.

### 5.2 Table layouts

Tabular files (text, Excel, Parquet, Feather, SQLite) can hold spectra in three
layouts. `layout="auto"` (the default) recognises them:

| Layout | Shape | Example |
|---|---|---|
| `wide` / `columns` | one spectrum per column; a monotonic first column is the axis | `shift, sample_1, sample_2` |
| `rows` | one spectrum per row; a monotonic first row (or numeric header) is the axis; leading cells of that row may be empty for position columns | LabSpec map export |
| `long` (tidy) | one point per row: `[spectrum,] [name,] [x, y, z,] wavenumber, intensity` | Renishaw WiRE map export, files written by RADAR |

Column names are matched case-insensitively. The axis column can be called
`wavenumber`, `raman shift`, `shift`, `wavelength`, `#wave`, `x-axis`, ...;
the intensity column `intensity`, `counts`, `#intensity`, `signal`, `y`, ...;
position columns `x`, `y`, `z`, `#x`, `#y`, `time`, `depth`. Pass
`layout="wide"`, `"rows"` or `"long"` when the guess is wrong.

### 5.3 Instrument text exports

| Export | Recognised by |
|---|---|
| Renishaw WiRE `.txt` (`wl int`, `x wl int`, `y x wl int`, with or without `#X #Y #Wave #Intensity` header) | repeating axis column → long layout; positions kept |
| Horiba LabSpec `.txt` (single spectra and maps: `<tab><tab>w1 w2 …`, then `x y i1 i2 …`) | `#key=value` header skipped; leading empty cells → position columns |
| Ocean Insight / OceanView `.txt` | data between `>>>>>Begin Spectral Data<<<<<` and `>>>>>End…` |
| RRUFF `.txt` / `.rruff` | `##` header lines and `##END=` skipped |
| WITec, Wasatch Enlighten, B&W Tek and other two-column or table exports | generic text rules |
| European exports (`;` separator, `1234,5`) | decimal comma detected (`decimal=","` to force) |

Other options for text files: `delimiter=`, `decimal=`, `comments="#"`,
`skip_header=`, `encoding=`.

### 5.4 The `SpectralData` object

```python
data = radar.read("map.wdf")
data.intensities      # (n, length) array, or a list of arrays if lengths differ
data.wavenumbers      # (length,) axis, or None
data.x_units          # e.g. "1/cm", "nm", "Raman shift (cm-1)"
data.names            # spectrum names, or None
data.positions        # (n, k) coordinates for maps / scans / series, or None
data.position_names   # e.g. ["x", "y"]
data.map_shape        # e.g. (20, 20) for a regular map, else None
data.as_map()         # intensities reshaped to (*map_shape, length)
data.meta             # instrument metadata (format, title, laser, comments, ...)
len(data), data.length, data.ragged
```

`map_shape` comes from the file (WDF, LabSpec, TriVista) or is inferred when
the x/y positions form a complete grid.

You can also build `SpectralData` from arrays:

```python
data = radar.SpectralData.from_arrays(X, wavenumbers=wn, names=names)   # X: (n, L) or (ny, nx, L)
```

### 5.5 Your own arrays

You never have to use `read`: every function accepts NumPy arrays directly.
A 1-D array is one spectrum, a 2-D array is `(n_spectra, length)`, a 3-D
array is a map `(ny, nx, length)`, and a list of 1-D arrays may hold spectra
of different lengths.

## 6. Decomposing spectra

```python
comps = radar.decompose(spectra)                          # pre-trained ensemble model
comps = radar.decompose(spectra, model="standard")
comps = radar.decompose("data.csv", output="result.xlsx") # read, decompose, save

model = radar.load_model("ensemble")                      # keep a model for repeated use
comps = model.decompose(spectra)
comps = model(spectra)                                    # same
```

| Argument | Default | Meaning |
|---|---|---|
| `spectra` | | array, list of arrays, `SpectralData` or file path |
| `wavenumbers` | `None` | spectral axis to store in the result (the network does not use it) |
| `normalize` | `True` | min-max scale each spectrum to [0, 1] first; set `False` only for data already in [0, 1] |
| `units` | `"original"` | `"original"`: components in input units (they add up to the input); `"normalized"`: the [0, 1] network scale |
| `batch_size` | `32` | spectra per network call |
| `output` | `None` | also save the result to this file |
| `verbose` | `0` | print progress |
| other keywords | | passed to `radar.read` when `spectra` is a path (`layout`, `key`, `sheet`, `query`, ...) |

**How intensities are handled.** Each spectrum *x* is scaled to
`(x − min) / (max − min)`, decomposed, and the components are scaled back. All
components are multiplied by `(max − min)`, and `min` is added to the
baseline. The result therefore does not depend on the intensity scale or
offset of the input (up to float32 rounding).

**Which model?**
- `ensemble` is the default.
- `standard` gives an exactly additive decomposition, with noise = input −
  baseline − cosmic rays − peaks.
- On synthetic test data the two are close; see the table in §13. Try both on
  a few of your spectra and inspect the plots.

**Maps, line scans, series.** Positions and map shape are carried from the
input to the result, so `comps.as_map("peaks")` and `comps.band_image(...)`
work (§7.2).

**Spectra of different lengths** (a list, a ragged SPC or JCAMP file) return
a list of `Components`, one per spectrum.

## 7. Working with the results

### 7.1 `Components`

```python
comps.spectra         # the input
comps.baseline, comps.cosmic_rays, comps.noise, comps.peaks
comps.wavenumbers, comps.names, comps.positions, comps.map_shape, comps.x_units
comps.units           # "original" or "normalized"
comps.model           # "ensemble", "standard" or "custom"
comps.meta            # per-spectrum normalisation (offset, scale)
len(comps), comps.length
comps[0], comps[10:20]   # select spectra
```

Derived spectra:

| Property | Definition | Use |
|---|---|---|
| `corrected` | `peaks` | clean spectra for analysis |
| `denoised` | `spectra − noise` | keep baseline and peaks, remove noise |
| `despiked` | `spectra − cosmic_rays` | cosmic-ray removal only |
| `baseline_corrected` | `spectra − baseline` | background removal only |
| `reconstruction` | sum of the four components | |
| `residual` | `spectra − reconstruction` | 0 for `standard`; small for `ensemble` |

### 7.2 Maps

```python
comps.as_map("peaks")              # (ny, nx, length)
comps.as_map("baseline")
img = comps.band_image(990, 1010)  # integrated peak area between two wavenumbers -> (ny, nx)
```

`band_image` integrates any component (`component="peaks"` by default)
between two axis values. Without a map it returns one value per spectrum.

### 7.3 Plots

```python
fig = comps.plot(index=0)                        # input + 4 components, 5 panels
from radar.plotting import plot_comparison
fig = plot_comparison(test_set, comps, index=0)  # true vs predicted (synthetic data)
```

## 8. Saving results and converting files

```python
comps.save("result.h5")                 # same as radar.write(comps, "result.h5")
radar.write(comps, "result.xlsx")
radar.write(data, "spectra.jdx")        # SpectralData
radar.write(X, "spectra.csv", wavenumbers=wn, names=names)   # plain arrays
radar.write(test_set, "test.parquet")   # SyntheticDataset
```

How each format stores a decomposition (spectra + 4 components):

| Format | Layout |
|---|---|
| `.npz`, `.h5`, `.nc`, `.mat`, `.json` | one array per component, plus axis, names, positions and metadata |
| `.csv`, `.tsv`, `.txt`, `.parquet`, `.feather`, `.db` | one tidy table: `spectrum, [name], [x, y], wavenumber, spectra, baseline, cosmic_rays, noise, peaks` |
| `.xlsx` | one sheet per component (spectra in columns), or one tidy sheet for maps |
| `.spc`, `.jdx`, `.npy` | one file per component: `result_spectra.spc`, `result_baseline.spc`, ...; `write` returns the list |

Plain spectra are written in the *wide* layout to text and Excel, and in the
*long* layout to Parquet, Feather and SQLite. Override this with
`layout="wide"` or `layout="long"`. Other options: `decimal=","` for text,
`table=` and `if_exists="append"` for SQLite.

**Reading results back:**

```python
comps = radar.load_components("result.h5")          # any multi-array format
comps = radar.load_components("result_peaks.spc")   # finds the sibling files
```

**Converting between formats:**

```python
radar.write(radar.read("old.spc"), "new.jdx")
```

From the command line: `radar convert old.spc new.jdx`.

## 9. Models

### 9.1 Loading and saving

```python
model = radar.load_model("ensemble")                         # pre-trained
model = radar.load_model("my_model.keras")                   # saved with model.save
model = radar.load_model("w.weights.h5", architecture="standard")
model = radar.load_model("Models/ensembleModelRes.h5", architecture="ensemble")  # original weight files
model.save("my_model.keras")       # architecture + weights + optimiser state
model.save("my_model.weights.h5")  # weights + architecture name (no optimiser state)
model.summary()                    # layer table
model.model                        # the underlying keras.Model
```

### 9.2 Which software versions does a saved model need?

Every model saved by RADAR records how it was made:
- the versions of **RADAR, Keras, TensorFlow, Python, NumPy and h5py**, and
  the platform;
- the architecture and its hyper-parameters;
- the training settings, including the number of spectra seen and, for a
  fine-tuned model, the model it started from;
- the date it was saved.

Ask for it without loading the model:

```python
radar.model_info("my_model.keras", show=True)
```

```
architecture : standard (scaler=20, dim_red_size=256, l1_param=1e-06, l2_param=1e-06)
parameters   : 585,299
origin       : bundled pre-trained weights (standard.h5)
saved        : 2026-10-24T04:35:03+00:00
saved with   : radar 0.1.0, keras 3.15.1, keras_backend tensorflow, tensorflow 2.21.0, python 3.13.17, numpy 2.5.3, h5py 3.14.0, platform Linux x86_64
training     : epochs=12, steps_per_epoch=200000, batch_size=100, spectra_seen=240000000, ...
compatibility: loads with the installed versions
```

The same is available from the command line (`radar info my_model.keras`),
and as a dict (`radar.model_info(path)["saved_with"]["tensorflow"]`).

**Where it is stored.**
- `.keras` files: an extra `radar_info.json` entry in the archive. A `.keras`
  file is a zip, so `unzip -p my_model.keras radar_info.json` shows it even
  without RADAR.
- `.weights.h5` files: a `radar_info` attribute on the HDF5 root (`h5dump -A`
  or h5py).

Because the architecture is recorded, `.weights.h5` files no longer need an
`architecture=` argument.

**Loading with other versions.**

| Situation | What happens |
|---|---|
| saved with the same or an older Keras / TensorFlow | loads normally |
| saved with a newer Keras / TensorFlow | a warning names the version to install, e.g. ``saved with keras 3.15.1, installed 3.12.4: upgrade with `pip install -U "keras>=3.15.1"` ``; RADAR still loads the weights by rebuilding the network (the optimiser state is then not restored, the predictions are identical) |
| Keras 2 weight file (original scripts) | loads with `architecture=`; `model_info` reports the Keras 2 origin |
| `.keras` file saved by plain Keras, without RADAR | `model_info` reports the Keras version from the archive's `metadata.json` |

Newer Keras 3 versions are designed to read files written by older ones, but
not the other way round. Saving with a recent version is therefore never a
problem for you, but it can be for a colleague with older software. To share
a model, tell them the versions from `model_info`. Alternatively, share the
`.weights.h5` file: it holds only weights and is the most portable format.

For full reproducibility of an analysis, also record the environment the
results were produced with: `radar info` prints the installed versions.

### 9.3 Architectures

See [Models](models.md) for the layer-by-layer description. Both networks can
be built with other sizes, for example
`radar.new_model("ensemble", scaler=16, dim_red_size=196)`, but the
pre-trained weights only fit the default sizes.

## 10. Training a new model

```python
model = radar.train(
    model="ensemble",                  # or "standard"
    epochs=12, steps_per_epoch=200_000, batch_size=100,   # 240 M spectra, as the pre-trained models
    length=750, min_width=5, max_width=256, max_num_peaks=5,
    learning_rate="original",          # 12-level decay 1.05e-3 -> 1.5e-6
    checkpoint_path="best.keras",      # keep the best model (validation loss)
    save_to="my_model.keras",
)
model.history.history["loss"]
```

| Option | Default | Meaning |
|---|---|---|
| `data` | `None` | `None`: fresh synthetic spectra every batch; or a `SyntheticDataset`, `(X, y)` arrays, or a data set file |
| `epochs`, `steps_per_epoch`, `batch_size` | 10, 1000, 100 | `steps_per_epoch` only applies to the synthetic stream |
| `learning_rate` | `"original"` | the original 12-level schedule spread over the run; or a float / Keras schedule |
| `length`, `min_width`, `max_width`, `max_num_peaks`, `increase_noise` | 750, 5, 256, 5, False | synthetic training spectra |
| `legacy_scaling` | `False` | reproduce the input scaling of the original generator exactly (§12) |
| `validation_data` | `"synthetic"` | 1000 spectra of 2000 points; `None`, a data set or `(X, y)` |
| `checkpoint_path`, `callbacks`, `seed`, `verbose` | | as in Keras |

`y` for your own arrays is `[baseline, cosmic_rays, noise, peaks]` (or a
dict with those keys), each of shape `(n, length)`.

**Cost.** The full schedule processes 240 million spectra; plan for days on a
GPU. On a CPU a step of 100 spectra takes about 3–7 s. Use short runs only to
test a set-up.

The lower-level `Decomposer` methods do the same thing:
`radar.new_model("ensemble").train(...)`.

## 11. Fine-tuning a model

To adapt the pre-trained weights to an instrument, you need spectra whose
components are known. Usually these are synthetic spectra that resemble your
data (length, band widths, noise level).

```python
train, test = radar.train_test_datasets(
    5000, 500, length=1340,
    train_kwargs=dict(min_width=8, max_width=300),
    test_kwargs=dict(min_width=8, max_width=300),
)
before = radar.evaluate("ensemble", test)
model = radar.train(train, model="ensemble", init="pretrained", epochs=5, batch_size=32,
                    learning_rate=1e-5, validation_data=test, save_to="tuned.keras")
after = radar.evaluate(model, test)
```

- Use a **small constant learning rate** such as `1e-5`. The original
  schedule starts at 1e-3 and would overwrite the pre-trained weights.
- Always compare before and after on held-out data. With small data sets,
  fine-tuning can make results slightly *worse*: batch-normalisation
  statistics are re-estimated from a few batches. Keep the pre-trained model
  in that case.
- `init` can also be a saved model file, to continue training it.

## 12. Synthetic data

```python
test = radar.generate(1000, length=1024, seed=0)       # test settings (default)
train = radar.generate(5000, length=750, min_width=5, max_width=256,
                       max_num_peaks=5, increase_noise=False, seed=1)   # training settings
train, test = radar.train_test_datasets(5000, 1000, length=1024, seed=0) # independent streams
test.X, test.baseline, test.cosmic_rays, test.noise, test.peaks         # (n, length) each
X, y = test.as_keras()          # (n, L, 1), [4 × (n, L)] float32
test.save("test.h5")            # any multi-array format
test = radar.load_dataset("test.h5")
```

**What a synthetic spectrum contains.**
- **Baseline:** a random sum of linear, sinusoidal, polynomial, Gaussian and
  Lorentzian curves.
- **Cosmic rays:** 0–29 one-point spikes.
- **Noise:** Gaussian with a random σ.
- **Peaks:** 1 to `max_num_peaks − 1` Gaussian, Lorentzian or Voigt bands
  with widths in `[min_width, max_width)` points.

The three signal components get random weights. In every sample, the input
equals the sum of its four components.

| Setting | `increase_noise` | Scaling |
|---|---|---|
| training | `False` | divided by the maximum; noise on the scale of the peaks, absent in 5 % of samples |
| test (default of `generate`) | `True` | noise ×3–9 (lower SNR); min-max scaled to [0, 1] |

**`legacy_scaling=True`** reproduces a quirk of the original generator. In
its training setting, the targets were divided by the spectrum maximum but
the input was not. The pre-trained weights were trained that way. RADAR
scales consistently by default; see [differences](differences.md).

All functions take `seed=` (or `rng=`) and never use NumPy's global random
state. For training streams, use `radar.batch_generator(...)` (Python
generator) or `radar.make_tf_dataset(...)` (`tf.data`).

## 13. Evaluating a model

```python
res = radar.evaluate(model, test)                    # model: name, file or Decomposer
res["peaks"]   # {'rmse': ..., 'mae': ..., 'max_error': ..., 'r2': ...}
stats = radar.evaluate(model, test, statistics=True)
stats["noise"] # {'rmse_mean', 'rmse_median', 'rmse_variance',
               #  'max_error_mean', 'max_error_median', 'max_error_variance'}
```

| Metric | Per spectrum | Summary in `evaluate` |
|---|---|---|
| RMSE | √mean((true − pred)²) | mean over spectra |
| MAE | mean \|true − pred\| | mean over spectra |
| **Max error** | max \|true − pred\|, the worst single point | mean over spectra |
| R² | — | pooled over all points of all spectra |

Max error complements RMSE: RMSE averages over the whole spectrum, while max
error shows the largest local mistake, such as a missed cosmic ray, a clipped
peak top or a baseline bump. R² is pooled because it is undefined for a
spectrum whose true component is constant (e.g. no cosmic rays).
`statistics=True` gives the mean, median and variance over spectra of the
per-spectrum RMSE and max error.

Low-level functions: `radar.metrics.rmse`, `mae`, `max_error`, `r2`
(per spectrum), `per_spectrum_errors`, `error_statistics`,
`evaluate_components` (any two objects or dicts with the four components),
and `snr`.

**Reference values.** Pre-trained models on 500 synthetic test spectra
(`generate(500, length=1024, seed=2024)`, normalised scale; from
`examples/06_compare_models.py`):

| Component | Model | RMSE | MAE | Max error | R² |
|---|---|---|---|---|---|
| baseline | standard | 0.0213 | 0.0158 | 0.0546 | 0.976 |
| baseline | ensemble | 0.0306 | 0.0244 | 0.0763 | 0.941 |
| cosmic rays | standard | 0.0108 | 0.0012 | 0.1647 | 0.814 |
| cosmic rays | ensemble | 0.0106 | 0.0012 | 0.1565 | 0.800 |
| noise | standard | 0.0186 | 0.0119 | 0.1682 | 0.944 |
| noise | ensemble | 0.0176 | 0.0109 | 0.1677 | 0.945 |
| peaks | standard | 0.0239 | 0.0144 | 0.0936 | 0.899 |
| peaks | ensemble | 0.0245 | 0.0145 | 0.1020 | 0.880 |

These are synthetic test data from this package's generator. Use them to see
the size of the errors, not as a benchmark for measured spectra.

## 14. Command line

| Command | Does |
|---|---|
| `radar decompose IN [IN ...] -o OUT [-m ensemble\|standard\|FILE]` | decompose files; with several inputs `-o` is a folder or a pattern with `{stem}` |
| `radar convert IN OUT` | convert spectra or a decomposition between formats |
| `radar generate -n N -o OUT [--length L] [--train]` | synthetic data set |
| `radar train -o MODEL.keras [-a ensemble] [--init pretrained\|FILE] [--data FILE]` | train or fine-tune |
| `radar evaluate DATA [-m MODEL] [--statistics]` | metrics as JSON |
| `radar formats` / `radar info [MODEL]` | formats / installation, or the versions a model was saved with |

```bash
radar decompose sample.wdf -o sample_components.h5
radar decompose data/*.spc -o "results/{stem}_components.csv" -m standard
radar decompose export.csv -o out.db --layout long --decimal ,
radar convert measurement.spc measurement.jdx
radar generate -n 1000 -o test.h5
radar evaluate test.h5 -m tuned.keras --statistics
```

All read options (`--format`, `--layout`, `--key`, `--sheet`, `--table`,
`--query`, `--delimiter`, `--decimal`) and `--out-layout` are available. See
[Command line](cli.md).

## 15. Practical advice and limitations

- **Band width in points matters.** The networks work on point indices, not
  on wavenumbers. They were trained on 750-point spectra with bands 5–256
  points wide. If your bands are much narrower or wider in points (very fine
  or very coarse sampling), consider resampling or fine-tuning.
- **Look before you batch.** Plot a few decompositions (`comps.plot(i)`)
  before processing a whole data set. Very sharp, intense bands can be partly
  assigned to cosmic rays, and structured noise can leak into the peaks.
- **Normalisation is per spectrum.** A huge cosmic ray compresses the rest of
  that spectrum into a small part of [0, 1]. The networks are trained for
  this, but precision drops.
- **Axis direction.** Files that store the axis from high to low wavenumber
  are processed in file order. The components are returned in the same order
  as the input.
- **The ensemble residual is not zero.** If you need an exactly additive
  decomposition, use `standard`, or redefine the noise as
  `spectra − baseline − cosmic_rays − peaks`.
- **NaN values** are rejected. Interpolate or crop them first.

## 16. Troubleshooting

| Problem | Fix |
|---|---|
| `ImportError: ... pandas` / `pyarrow` / `openpyxl` | `pip install "radar-raman[formats]"` |
| `ImportError: ... renishawWiRE` | `pip install renishawWiRE` |
| `.xml` / `.tvf` files | `pip install rosettasciio` (GPL-3.0) |
| `cannot tell the format of ...` | pass `format="csv"` (or another name from `radar.formats()`) |
| a table is read with the wrong orientation | pass `layout="wide"`, `"rows"` or `"long"` |
| numbers with decimal commas read as text | `decimal=","` |
| `UserWarning: ... saved with keras X, installed Y` | install the version shown, or ignore: the weights are still loaded (§9.2) |
| `RuntimeWarning: ... saved with a newer Keras version` | harmless: the weights were loaded (§9.2) |
| which versions does a model file need? | `radar.model_info(path, show=True)` or `radar info path` (§9.2) |
| `ValueError: spectra contain NaN` | clean the data first |
| out of memory with very large maps | decompose in chunks: `model.decompose(X[i:i+10000])`, or lower `batch_size` |
| slow training | use a GPU; on CPU, train only short test runs |

More detail: [Models](models.md) · [Synthetic data](data_generation.md) ·
[Command line](cli.md) · [API reference](api.md) ·
[Differences from the original code](differences.md)
