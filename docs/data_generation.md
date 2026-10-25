# Synthetic data

The networks are trained on synthetic spectra whose components are known
exactly. The generator (`radar.generator`) reproduces the original one.

## Components

| Component | How it is drawn |
|---|---|
| baseline | A random subset of five curves on [0, 1]: linear; sinusoidal (random frequency and phase); polynomial of degree 2–10; a Gaussian with its centre near one end; a Lorentzian with its centre near one end. Each curve is normalised, scaled and shifted randomly. |
| cosmic rays | 0–29 single-pixel spikes with heights uniform in [0, 1) |
| noise | white Gaussian noise with σ ~ U(0, 0.1); the end points are set to 0 |
| peaks | 1 to `max_num_peaks − 1` bands of width `[min_width, max_width)` points. Each band is Gaussian, Lorentzian or Voigt (equal probability) with a random height. The result is normalised to a maximum of 1. |

`generate_sample` then weights the baseline, cosmic rays and peaks with
random proportions that sum to 1, and scales the spectrum:

- **Training setting** (`increase_noise=False`): the spectrum is divided by
  its maximum, the noise takes the scale of the peaks, and in 5 % of the
  samples there is no noise.
- **Test setting** (`increase_noise=True`): the noise is amplified 3–9×
  (lower SNR), and the spectrum is min-max normalised to [0, 1], with the
  baseline absorbing the shift.

In both settings `spectrum == baseline + cosmic_rays + noise + peaks`. For
the `legacy_scaling` option, see [differences](differences.md).

## API

```python
import radar

# finite data sets -> SyntheticDataset(X, baseline, cosmic_rays, noise, peaks, params)
test = radar.generate_dataset(1000, length=1024, min_width=10, max_width=300,
                              increase_noise=True, seed=0)     # = original test data
train, test = radar.train_test_datasets(5000, 1000, length=750, seed=0)

X, y = test.as_keras()          # (n, L, 1), [4 x (n, L)] float32
test.subset(slice(0, 10))
test.save("test.h5"); radar.load_dataset("test.h5")   # any multi-array format

# endless batches for training
gen = radar.batch_generator(length=750, batch_size=100, seed=0)
ds  = radar.make_tf_dataset(length=750, batch_size=100, seed=0)   # tf.data version

# single spectra / components
s, targets = radar.generate_sample(1024, rng=0)
from radar import generator
generator.generate_baseline(1024, rng=0)
generator.generate_peaks(1024, max_num_peaks=6, min_peak_width=10, max_peak_width=120, rng=0)
generator.generate_cosmic_rays(1024, max_num=10, rng=0)
generator.generate_noise(1024, scaling_factor=0.1, rng=0)
```

**Reproducibility.** Every function takes a seed or a
`numpy.random.Generator` (`seed=` / `rng=`) and never touches NumPy's global
random state, so the same seed always gives the same data.
`train_test_datasets` draws its two sets from independent streams.

**Defaults.**

- `generate_dataset` (alias `radar.generate`) uses the original test settings: widths 10–300, increased
  noise.
- `batch_generator` and `Decomposer.train` use the training settings: length
  750, widths 5–256, `max_num_peaks=5`, normal noise.
- For spectra shorter than 500 points, the peak widths are set to
  `[length/32, length/2)`, as in the original code.
