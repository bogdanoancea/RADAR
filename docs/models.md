# Models

Both networks map a batch of normalised spectra `(batch, length, 1)` to four
outputs `(batch, length)`, always in the order
`[baseline, cosmic_rays, noise, peaks]` (`radar.COMPONENTS`). Every
convolution uses "valid" padding on an edge-padded input (`EdgePad1D`), so
the spectrum length is preserved for any input length, and the spectrum
edges are extended with their own values instead of zeros.

## Building blocks

**Key-vector block.** A 1-D convolution with `dim_red_size` filters is
followed by global max pooling, which gives a learned *key* vector of length
`D` that summarises the whole spectrum. The outer product of the signal
`(L × 1)` with the key `(D × 1)` gives an `L × D` map. Two 2-D convolutions
process this map with kernels of (32, D/10) and (16, D/10) and strides of
(1, √D), reducing it back to `L × scaler` features. This gives each position
access to global information about the spectrum.

**Decoder.** 1-D convolutions with kernels of 128 and 64, each followed by
batch normalisation and LeakyReLU (slope 0.3).

**Heads.** Each component has its own output head:

| Head | Layers | Rationale |
|---|---|---|
| baseline | Conv1D(k=64) → average pooling (33) → ReLU | wide and smooth |
| cosmic rays | Conv1D(k=3) → ReLU | 1–2 pixel spikes |
| peaks | Conv1D(k=8) → average pooling (4) → ReLU | narrow kernel keeps sharp bands |
| noise | *standard*: input − other three; *ensemble*: Conv1D(k=1) → tanh | |

Regularisation: L1/L2 = 1e-6 on hidden kernels, L1 = 1e-5 on head kernels,
and L2 = 1e-6 on biases and activities.

## `standard`

`build_standard_model(scaler=20, dim_red_size=256)` has **585 299**
parameters. It has one encoder, one key-vector block and one decoder, with
heads for the baseline, cosmic rays and peaks. The noise is the remainder
`input − baseline − cosmic rays − peaks`, so the four components always sum
to the input. The loss is RMSE.

## `ensemble`

`build_ensemble_model(scaler=10, dim_red_size=144)` has **280 592**
parameters and works in two stages:

1. **Stage 1** (same structure as `standard`) predicts the baseline and the
   cosmic rays.
2. The *reduced spectrum*, `input − baseline − cosmic rays`, goes through
   **stage 2**: a second encoder and key-vector block, with a residual
   connection that adds stage 1's 2-D features. Stage 2 has heads for the
   peaks and the noise.

The noise is predicted directly (a tanh output), so the components only sum
to the input approximately (`Components.residual`). The loss is
RMSE + (1 − R²).

## Custom hyper-parameters

```python
radar.Decomposer.new("ensemble", scaler=16, dim_red_size=14**2, l1_param=1e-6, l2_param=1e-6, seed=1)
keras_model = radar.build_model("standard", scaler=32)   # raw, uncompiled Keras model
```

`dim_red_size` must be a perfect square. The pre-trained weights only fit
the default hyper-parameters.

## Pre-trained weights

| Architecture | File in package | Original file | SHA-256 |
|---|---|---|---|
| standard | `radar/weights/standard.h5` | `Models/standardModel.h5` | `9eecea6445d2ab0d952f6eba19a10f3a182d3262b60c77228196234dd1b326b7` |
| ensemble | `radar/weights/ensemble.h5` | `Models/ensembleModelRes.h5` | `69e741d53dee6ab8b8de7cf515c9ea5c6a01534f4ad1f7642a8b7e8a21853039` |

The files are the unmodified Keras 2.7 weight files from the original
repository, trained by J. Sjöberg. Each model saw
**240 million** freshly generated synthetic spectra of 750 points: 12 epochs
× 200 000 steps × 100 spectra per batch, with the piecewise-constant
learning-rate decay from 1.05e-3 to 1.5e-6. Keras 2 names
layers automatically (`conv1d_4`, `batch_normalization_7`, …) and loads
weights by layer order. RADAR names every layer explicitly and loads the old
files through a fixed name mapping (`radar.models.LEGACY_LAYER_NAMES`),
checking every weight shape.

### Validation of the port

The original code (`Scripts/essentials.py`) was run under legacy Keras 2
(`tf-keras`) with the original weight files on two inputs: six synthetic test
spectra of length 1024, and three spectra of length 1738. Its outputs are
stored in `tests/data/reference_outputs.npz`. The RADAR implementation
reproduces all four outputs of both models to a maximum absolute difference
of about 1e-6 (float32 rounding). The test `tests/test_models.py::test_matches_original_implementation`
checks this with a tolerance of 1e-5.
