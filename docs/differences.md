# Differences from the original scripts

RADAR is based on the original repository
([JoelSjoberg/Comprehensive-Raman-Spectral-Analysis-Using-Deep-Learning](https://github.com/JoelSjoberg/Comprehensive-Raman-Spectral-Analysis-Using-Deep-Learning)):
`Scripts/essentials.py`, `Scripts/generator.py`, `TrainStandard.py`,
`TrainEnsemble.py` and `GenerateTestData.ipynb`. The **network architectures
and pre-trained weights are unchanged**, and their outputs are reproduced to
float32 precision (see [Models](models.md#validation-of-the-port)). This page
lists every intentional change.

## Framework

- **Keras 3.** The original code targets Keras 2 (`tf.keras` up to TF 2.15).
  It applies raw TensorFlow operations to symbolic tensors
  (`tf.keras.backend.expand_dims`, unary minus, `tf.nn.relu`), which Keras 3
  does not allow. These became Keras layers (`ExpandLastDim`,
  `ResidualSubtract`, `Activation`), so the models can be saved and loaded as
  `.keras` files. The computation is the same.
- **Layer names.** Every layer has a descriptive name. The old Keras 2 weight
  files load through an explicit name mapping instead of relying on layer
  order.
- **`CustomPad.get_config`** stored half the pad size, so a model rebuilt from
  its config padded incorrectly. `EdgePad1D` stores the true size;
  `CustomPad` remains as an alias.
- The loss functions use `keras.ops` instead of `keras.backend`, with the
  same formulas.

## Data generator

- **Random state.** The original functions use NumPy's global random state.
  RADAR functions take a seed or `numpy.random.Generator`, so data sets are
  reproducible and independent of other code. The distributions are the
  same, but a given seed gives different draws than `np.random.seed` did in
  the original.
- **Input scaling in the training setting** (`increase_noise=False`). In the
  original `train_generator`, the baseline, cosmic-ray and peak *targets*
  are divided by the spectrum maximum, but the *input* spectrum is not:

  ```python
  s = bl + ray + peak
  maxim = np.max(s)
  bl = bl/maxim; ray = ray/maxim; peak = peak/maxim
  ...
  s = s + noise          # s was not divided by maxim
  ```

  The training inputs therefore did not equal the sum of their targets. On
  200 samples the median maximum discrepancy was 0.25, for inputs with a
  median maximum of 0.89. The original comment ("The spectrum is the sum of
  the 4 components") and the test setting show the intent: the input should
  be the sum. RADAR scales the input consistently by default.
  `legacy_scaling=True` (in `generate_sample`, `generate_dataset`,
  `batch_generator` and `Decomposer.train`) reproduces the original behaviour
  exactly. **The pre-trained weights were trained with the original
  behaviour**, which matters if you want to replicate that training run.
  The test setting (`increase_noise=True`) was already consistent and is
  unchanged.
- **All-zero samples.** The three component weights are drawn from the
  integers 0–9. If all three were 0, or the weighted baseline, cosmic rays
  and peaks were all zero (for example, all the weight on cosmic rays but no
  ray drawn), the original code divided 0 by 0. `nan_to_num` then turned the
  sample into zeros. RADAR draws such samples again.
- **Very narrow Lorentzian bands** (1–2 points wide) divided 0 by 0 in the
  original `lorentz`, which turned the whole peak spectrum into zeros. RADAR
  returns a flat unit band instead. This only happens for very short spectra
  or a `min_width` below 3.
- `generate_peaks` with `max_num_peaks <= 1` raised an error in the original;
  it now draws one band.

## Training scripts

- `TrainStandard.py` calls
  `train_generator(..., max_num_bands=5)`, but the generator has no
  `max_num_bands` parameter, so the script as committed raises a
  `TypeError`. The standard model was presumably trained with an earlier
  version of the generator. RADAR uses `max_num_peaks=5` for both models, as
  in `TrainEnsemble.py`.
- The learning-rate schedule is the same (12 levels × 1.5, boundaries every
  200 000 steps). `original_learning_rate_schedule(total_steps)` stretches it
  to any run length; `learning_rate="original"` uses it.
- The validation set is generated the same way (1000 spectra of 2000 points,
  widths 5–300, `max_num_peaks=5`).

## New in the package

The following did not exist as reusable code in the original repository:

- the simple API (`radar.decompose`, `read`, `write`, `load_model`, `train`,
  `generate`, `evaluate`) and the `Decomposer` class (normalisation, mapping
  back to input units, maps, spectra of different lengths);
- file I/O for SPC, JCAMP-DX, Renishaw WDF, Horiba LabSpec XML, TriVista TVF,
  instrument text exports, Excel, Parquet, Feather, SQLite, HDF5 / NetCDF,
  MATLAB, JSON and NumPy;
- saving and loading models in `.keras` / `.weights.h5` format;
- the command-line tool, metrics (including max error and the
  mean / median / variance summaries used in the original evaluation
  notebooks), plotting, tests and documentation. Comparison baselines from the evaluation
notebooks (Savitzky–Golay, ORPL bubble fill, the Kazemzadeh et al. and
Barton et al. models) and the saliency analyses are **not** included.
