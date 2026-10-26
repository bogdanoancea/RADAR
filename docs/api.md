# API reference

## The simple API

| Function | Description |
|---|---|
| `radar_raman.decompose(spectra, model="ensemble", output=None, **kw)` | decompose an array, `SpectralData` or file → `Components` (or a list for spectra of different lengths); `output=` saves the result |
| `radar_raman.read(path, format=None, **options)` | read spectra from any supported file → `SpectralData` |
| `radar_raman.write(obj, path, format=None, wavenumbers=None, names=None, **options)` | write spectra, `Components` or `SyntheticDataset` → path (or list of paths) |
| `radar_raman.formats(show=False)` | supported formats (list of dicts; prints a table with `show=True`) |
| `radar_raman.load_model(model="ensemble", architecture=None)` | pre-trained model by name, or a saved `.keras` / `.weights.h5` / Keras 2 `.h5` file → `Decomposer` |
| `radar_raman.train(data=None, *, model="ensemble", init=None, save_to=None, **train_options)` | train a new network or fine-tune one → `Decomposer` |
| `radar_raman.generate(n, length=1024, ...)` | synthetic spectra with known components → `SyntheticDataset` (alias of `generate_dataset`) |
| `radar_raman.evaluate(model, dataset, statistics=False)` | per-component metrics; `dataset` may be a file |

Also available: `radar_raman.model_info(model, show=False)` (software versions,
architecture and training settings recorded in a saved model),
`radar_raman.load_pretrained(architecture)`,
`radar_raman.new_model(architecture, **hyper)`, `radar_raman.load_components(path)`,
`radar_raman.save_components(c, path)`, `radar_raman.load_dataset(path)`,
`radar_raman.save_dataset(ds, path)`, `radar_raman.read_spectra` (alias of `read`) and
`radar_raman.save_spectra(path, X, wavenumbers, names)`.

## Classes

### `Decomposer`

| Member | Description |
|---|---|
| `Decomposer.pretrained(architecture)`, `.new(architecture, learning_rate=1e-5, **hyper)`, `.load(path, architecture=None)` | constructors |
| `.decompose(spectra, wavenumbers=None, normalize=True, units="original", batch_size=32, verbose=0, output=None, **read_options)` | → `Components`; calling the object does the same |
| `.predict_normalized(spectra)` | raw network output for [0, 1] spectra → dict |
| `.train(data=None, *, epochs, steps_per_epoch, batch_size, learning_rate="original", length, min_width, max_width, max_num_peaks, increase_noise, legacy_scaling, validation_data, validation_size, validation_length, seed, callbacks, checkpoint_path, verbose)` | train / fine-tune → Keras `History` |
| `.evaluate(dataset, normalize=False, statistics=False)` | metrics dict |
| `.save(path)` | `.keras` or `.weights.h5`; both record the software versions, architecture and training settings |
| `.info()` | provenance of the model plus the versions it runs with now |
| `.model`, `.architecture`, `.history`, `.training_info`, `.origin`, `.n_parameters`, `.summary()` | |

### `Components`

| Member | Description |
|---|---|
| `spectra`, `baseline`, `cosmic_rays`, `noise`, `peaks` | arrays `(n, length)` or `(length,)` |
| `wavenumbers`, `names`, `positions`, `position_names`, `map_shape`, `x_units`, `units`, `model`, `meta` | |
| `corrected`, `denoised`, `despiked`, `baseline_corrected`, `reconstruction`, `residual` | derived spectra |
| `as_map(component)`, `band_image(lo, hi, component="peaks")` | maps |
| `save(path)`, `plot(index)`, `to_dict()`, `to_record()`, `from_record(rec)`, `[i]`, `len()` | |

### `SpectralData`

| Member | Description |
|---|---|
| `intensities`, `wavenumbers`, `names`, `positions`, `position_names`, `map_shape`, `x_units`, `y_units`, `meta`, `source`, `arrays` | |
| `length`, `ragged`, `len()`, `as_map()` | |
| `SpectralData.from_arrays(intensities, wavenumbers=None, names=None, ...)` | build from arrays |

### `SyntheticDataset`

| Member | Description |
|---|---|
| `X`, `baseline`, `cosmic_rays`, `noise`, `peaks`, `params` | arrays `(n, length)` |
| `targets`, `as_keras()`, `subset(index)`, `length`, `len()` | |
| `save(path)`, `SyntheticDataset.load(path)` | any multi-array format |

## Modules

| Module | Contents |
|---|---|
| `radar_raman.models` | `build_model`, `build_standard_model`, `build_ensemble_model`, `compile_model`, `COMPONENTS`, `ARCHITECTURES`, `LEGACY_LAYER_NAMES` |
| `radar_raman.layers` | `EdgePad1D` (alias `CustomPad`), `ExpandLastDim`, `ResidualSubtract` |
| `radar_raman.losses` | `root_mean_squared_error`, `r_square`, `r_square_loss`, `joined_loss` |
| `radar_raman.generator` | `generate_dataset`, `train_test_datasets`, `generate_sample`, `generate_spectrum`, `batch_generator`, `make_tf_dataset`, `make_baselines`, `generate_baseline`, `generate_peaks`, `get_random_peak`, `generate_cosmic_rays`, `generate_noise` |
| `radar_raman.io` | `read`, `write`, `formats`, `SpectralData`, `Record`, `load_components`, `save_components`, `load_dataset`, `save_dataset` |
| `radar_raman.metrics` | `rmse`, `mae`, `max_error`, `r2` (per spectrum), `per_spectrum_errors`, `error_statistics`, `evaluate_components`, `snr` |
| `radar_raman.training` | `original_learning_rate_schedule`, `ORIGINAL_LR_VALUES`, `fit` |
| `radar_raman.preprocessing` | `minmax_normalize`, `denormalize_components`, `Scaling` |
| `radar_raman.plotting` | `plot_components`, `plot_comparison` |
| `radar_raman.cli` | `main`, `build_parser` |

---

## Docstrings

::: radar_raman.decomposer

::: radar_raman.components

::: radar_raman.io

::: radar_raman.models

::: radar_raman.generator

::: radar_raman.metrics

::: radar_raman.training

::: radar_raman.preprocessing

::: radar_raman.layers

::: radar_raman.plotting
