# Test data

`reference_outputs.npz` holds inputs and outputs of the **original**
implementation and is used by `tests/test_models.py` to check that the Keras 3
port in RADAR reproduces it exactly.

It was produced with the original repository code (`Scripts/essentials.py`,
`Scripts/generator.py`, `Models/*.h5`) running under legacy Keras 2
(`tf-keras 2.21`, `TF_USE_LEGACY_KERAS=1`):

- `X1024`: 6 synthetic test spectra (length 1024) from the original
  `train_generator(length=1024, batch_size=6, min_width=10, max_width=300,
  increase_noise=True)` with `np.random.seed(3)`;
- `X1738`: 3 normalised random-walk curves of length 1738 (the length of
  the live-cell spectra), to test another length;
- `{standard,ensemble}_{X1024,X1738}_{baseline,cosmic_rays,noise,peaks}`:
  predictions of `make_standard()` / `make_ensemble()` with the original
  weights loaded through `model.load_weights(...)`.
