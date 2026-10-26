# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""Synthetic Raman spectra with known components (training and test data).

A synthetic spectrum is the sum of four components, all sampled at random:

* **baseline** -- a random combination of linear, sinusoidal, polynomial,
  Gaussian and Lorentzian background curves;
* **cosmic rays** -- single-pixel spikes at random positions;
* **noise** -- white Gaussian noise with a random standard deviation;
* **peaks** -- Gaussian, Lorentzian and Voigt bands of random width,
  position and height (overlapping bands add up).

Because the ground-truth components are known, these spectra are used to
train the networks and to evaluate them.

The functions reproduce the generator of the original implementation
(J. Sjöberg). The differences are that every function takes a
``numpy.random.Generator`` (``rng``) instead of using the global NumPy state,
which makes data sets reproducible and independent of other code, and that
an all-zero component scaling (which produced an all-zero sample in the
original code) is re-drawn.

Main entry points
-----------------
:func:`generate_dataset`
    A finite data set (e.g. a test set) as a :class:`SyntheticDataset`.
:func:`train_test_datasets`
    Independent train and test sets.
:func:`batch_generator`
    An infinite Python generator of training batches (for ``model.fit``).
:func:`make_tf_dataset`
    The same batches wrapped in a ``tf.data.Dataset``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from scipy.special import wofz

from .models import COMPONENTS

__all__ = [
    "SyntheticDataset",
    "make_baselines",
    "generate_baseline",
    "get_random_peak",
    "generate_cosmic_rays",
    "generate_noise",
    "generate_peaks",
    "generate_spectrum",
    "generate_sample",
    "generate_dataset",
    "train_test_datasets",
    "batch_generator",
    "make_tf_dataset",
]

RngLike = Union[None, int, np.random.Generator]


def _rng(rng: RngLike) -> np.random.Generator:
    """Return a ``numpy.random.Generator`` from a seed, a Generator or None."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _normalize(x, eps=1e-11):
    """Min-max normalise ``x`` to [0, 1] (flat input -> zeros)."""
    return np.nan_to_num((x - np.min(x)) / (np.max(x) - np.min(x) + eps))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def make_baselines(
    length: int,
    sin_max_periodicity: float = 8,
    pol_max_degree: int = 10,
    eps: float = 1e-8,
    rng: RngLike = None,
) -> List[np.ndarray]:
    """Draw the five elementary baselines (each in [0, 1]).

    Returns
    -------
    list of ndarray
        ``[linear, sinusoidal, polynomial, gaussian, lorentzian]``, each of
        length ``length``.
    """
    rng = _rng(rng)
    sin_scaler, pol_scaler, gaussian_scaler, div_scaler = rng.random(4)
    x = np.linspace(0, 1, length)

    # Linear
    first_point, last_point = rng.random(2)
    lin = np.linspace(first_point, last_point, length)

    # Sinusoidal
    offset = rng.uniform(0, 2 * np.pi)
    stretch = rng.uniform(0.1, np.pi * sin_max_periodicity)
    sin = _normalize(np.sin(x * stretch + offset)) * sin_scaler
    sin = sin + rng.uniform(0, 1 - np.max(sin))

    # Polynomial
    degree = rng.integers(2, pol_max_degree + 1)
    pol = np.sum([rng.uniform(-1, 1) * np.power(x, i) for i in range(degree + 1)], axis=0)
    pol = _normalize(pol) * pol_scaler
    pol = pol + rng.uniform(0, 1 - np.max(pol))

    # Gaussian (centre close to one of the ends of the spectrum)
    a = rng.uniform(-1, 1)
    b = rng.uniform(0.8, 1) if rng.random() > 0.5 else rng.uniform(0, 0.2)
    c = rng.uniform(0.3, 1)
    gauss = a * np.exp(-np.power(x - b, 2) / (2 * c**2))
    gauss = _normalize(gauss) * gaussian_scaler
    gauss = gauss + rng.random() * (1 - np.max(gauss))

    # Lorentzian (centre close to one of the ends of the spectrum)
    a = rng.uniform(0.1, 0.5)
    if rng.random() > 0.5:
        a = -a
    b = rng.uniform(0.8, 1) if rng.random() > 0.5 else rng.uniform(0, 0.2)
    lor = (1 / np.pi) * a / ((a**2 + np.power(x - b, 2)) + eps)
    lor = _normalize(lor) * div_scaler
    lor = lor + rng.random() * (1 - np.max(lor))

    return [lin, sin, pol, gauss, lor]


def generate_baseline(length: int, rng: RngLike = None) -> np.ndarray:
    """A random baseline: the sum of a random subset of :func:`make_baselines`."""
    rng = _rng(rng)
    bls = make_baselines(length, rng=rng)
    mask = rng.integers(0, 2, size=len(bls))
    out = np.zeros(length)
    for m, b in zip(mask, bls):
        out += m * b
    return out


# ---------------------------------------------------------------------------
# Peaks
# ---------------------------------------------------------------------------
def _rbf(x, epsilon=1.0):
    return np.exp(-np.power(x * epsilon, 2))


def _lorentz(x, epsilon):
    curve = (1 / np.pi) * epsilon / (np.power(epsilon, 2) + np.power(x, 2))
    curve = curve - np.min(curve)
    peak = np.max(curve)
    # bands of 1-2 points are flat after the shift; the original code divided
    # 0/0 here, which turned the whole peak spectrum into zeros
    return curve / peak if peak > 0 else np.ones_like(curve)


def _voigt(x, sigma, gamma):
    z = (x + 1j * gamma) / (sigma * np.sqrt(2))
    profile = wofz(z).real / (sigma * np.sqrt(2 * np.pi))
    return _normalize(profile)


def get_random_peak(length: int = 16, extremity: float = 0.5, rng: RngLike = None) -> np.ndarray:
    """One band of ``length`` samples: Gaussian, Lorentzian or Voigt shape
    (equal probability) with a random height in [0, 1)."""
    rng = _rng(rng)
    line = np.linspace(-extremity, extremity, length)
    rbf_epsilon = rng.integers(5, 10)
    lorentz_epsilon = rng.uniform(0.1, 0.3)
    voigt_sigma = rng.uniform(0.01, 0.3)
    voigt_gamma = rng.uniform(0, 0.1)
    u = rng.random()
    if u < 0.33:
        p = _rbf(line, rbf_epsilon)
    elif u < 0.66:
        p = _lorentz(line, lorentz_epsilon)
    else:
        p = _voigt(line, voigt_sigma, voigt_gamma)
    return p * rng.random()


def generate_peaks(
    length: int = 1000,
    max_num_peaks: int = 10,
    min_peak_width: int = 16,
    max_peak_width: int = 100,
    eps: float = 1e-7,
    rng: RngLike = None,
) -> np.ndarray:
    """A peak-only spectrum, normalised to a maximum of 1.

    Draws between 1 and ``max_num_peaks - 1`` bands (upper bound exclusive,
    as in the original code) of width in ``[min_peak_width, max_peak_width)``
    at random positions. Overlapping bands add up.
    """
    rng = _rng(rng)
    spectrum = np.zeros(length)
    num_peaks = 1 if max_num_peaks <= 1 else rng.integers(1, max_num_peaks)
    for _ in range(num_peaks):
        width = int(rng.integers(min_peak_width, max_peak_width))
        mid = int(width / 2)
        loc = int(rng.integers(mid, max(length - mid, mid + 1)))
        peak = get_random_peak(length=width, rng=rng)
        seg = spectrum[loc - mid: loc - mid + width]
        spectrum[loc - mid: loc - mid + width] += peak[: len(seg)]
    spectrum[0] = 0
    spectrum[-1] = 0
    return spectrum / np.max(spectrum + eps)


# ---------------------------------------------------------------------------
# Cosmic rays and noise
# ---------------------------------------------------------------------------
def generate_cosmic_rays(length: int = 1000, max_num: int = 30, rng: RngLike = None) -> np.ndarray:
    """Between 0 and ``max_num - 1`` single-pixel spikes with heights in [0, 1)."""
    rng = _rng(rng)
    spectrum = np.zeros(length)
    n = rng.integers(0, max_num) if max_num > 0 else 0
    locs = rng.integers(0, length, size=n)
    spectrum[locs] = rng.uniform(0.0, 1.0, size=n)
    return spectrum


def generate_noise(length: int = 1000, scaling_factor: float = 0.1, rng: RngLike = None) -> np.ndarray:
    """White Gaussian noise with standard deviation ~ U(0, scaling_factor);
    the two end points are set to 0."""
    rng = _rng(rng)
    spectrum = rng.normal(0, rng.random() * scaling_factor, length)
    spectrum[0] = 0
    spectrum[-1] = 0
    return spectrum


# ---------------------------------------------------------------------------
# Full spectra
# ---------------------------------------------------------------------------
def generate_spectrum(
    length: int,
    min_width: int = 16,
    max_width: int = 256,
    num_peaks: int = 10,
    rng: RngLike = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw the four *unscaled* components of one spectrum.

    For ``length < 500`` the peak widths are set to ``[length/32, length/2)``
    (as in the original code).

    Returns
    -------
    tuple of ndarray
        ``(baseline, cosmic_rays, noise, peaks)``.
    """
    rng = _rng(rng)
    baseline = generate_baseline(length, rng=rng)
    rays = generate_cosmic_rays(length, rng=rng)
    noise = generate_noise(length, rng=rng)
    if length < 500:
        min_width = max(int(length / 32), 1)
        max_width = max(int(length / 2), min_width + 1)
    peaks = generate_peaks(
        length=length,
        max_num_peaks=num_peaks,
        min_peak_width=min_width,
        max_peak_width=max_width,
        rng=rng,
    )
    return baseline, rays, noise, peaks


def generate_sample(
    length: int = 1024,
    min_width: int = 5,
    max_width: int = 256,
    max_num_peaks: int = 10,
    increase_noise: bool = False,
    rng: RngLike = None,
    legacy_scaling: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """One training/test sample: an input spectrum and its four targets.

    The baseline, cosmic rays and peaks are weighted by random proportions,
    summed, and the spectrum is scaled to a maximum of ~1.

    * ``increase_noise=False`` (original training setting): the noise has
      the scale of the peaks and is dropped altogether in 5 % of the samples.
    * ``increase_noise=True`` (original test setting): the noise is
      amplified 3-9x (lower SNR) and the spectrum is min-max normalised to
      [0, 1]; the baseline absorbs the shift.

    In both cases ``spectrum == baseline + cosmic_rays + noise + peaks``.

    Parameters
    ----------
    legacy_scaling : bool
        Only relevant for ``increase_noise=False``. In the original training
        generator the targets (baseline, cosmic rays, peaks) were divided by
        the spectrum maximum but the input spectrum itself was not, so the
        input was *not* the sum of the targets. ``False`` (default) scales the
        input consistently; ``True`` reproduces the original behaviour exactly,
        e.g. to replicate the training of the published weights.

    Returns
    -------
    spectrum : ndarray, shape (length,)
    targets : dict
        Keys ``"baseline", "cosmic_rays", "noise", "peaks"``.
    """
    rng = _rng(rng)
    while True:
        bl, ray, noise, peak = generate_spectrum(length, min_width, max_width, max_num_peaks, rng=rng)
        scalers = rng.integers(0, 10, size=3).astype(float)
        if scalers.sum() == 0:
            continue
        scalers = scalers / scalers.sum()
        bl = bl * scalers[0]
        ray = ray * scalers[1]
        peak = peak * scalers[2]
        s = bl + ray + peak
        # The original code produced 0/0 (and then an all-zero sample) when the
        # weighted components were all zero, e.g. all weight on cosmic rays but
        # no ray drawn. Such samples are drawn again.
        if np.max(s) > 0:
            break

    if not increase_noise:
        maxim = np.max(s)
        bl, ray, peak = bl / maxim, ray / maxim, peak / maxim
        if rng.random() < 0.05:
            noise = np.zeros(length)
        noise = noise * scalers[2]
        s = (s if legacy_scaling else bl + ray + peak) + noise
    else:
        noise = noise * rng.integers(3, 10)
        noise = noise * scalers[2]
        s = s + noise
        minimum = np.min(s)
        s = s - minimum
        bl = bl - minimum
        maxim = np.max(s)
        s, bl, ray, peak, noise = (v / maxim for v in (s, bl, ray, peak, noise))

    out = [np.nan_to_num(v) for v in (s, bl, ray, noise, peak)]
    return out[0], dict(zip(COMPONENTS, (out[1], out[2], out[3], out[4])))


# ---------------------------------------------------------------------------
# Data sets
# ---------------------------------------------------------------------------
@dataclass
class SyntheticDataset:
    """A set of synthetic spectra with their ground-truth components.

    Attributes
    ----------
    X : ndarray, shape (n, length)
        Input spectra.
    baseline, cosmic_rays, noise, peaks : ndarray, shape (n, length)
        Ground-truth components (``X == sum of components``).
    params : dict
        Parameters used to generate the data set.
    """

    X: np.ndarray
    baseline: np.ndarray
    cosmic_rays: np.ndarray
    noise: np.ndarray
    peaks: np.ndarray
    params: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.X)

    @property
    def length(self) -> int:
        """Number of samples per spectrum."""
        return self.X.shape[1]

    @property
    def targets(self) -> List[np.ndarray]:
        """Targets in model-output order ``[baseline, cosmic_rays, noise, peaks]``."""
        return [self.baseline, self.cosmic_rays, self.noise, self.peaks]

    def as_keras(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """``(X[..., None], targets)`` -- ready for ``model.fit``/``evaluate``."""
        return self.X[..., None].astype("float32"), [t.astype("float32") for t in self.targets]

    def subset(self, index) -> "SyntheticDataset":
        """A new data set with the selected spectra (index, slice or mask)."""
        return SyntheticDataset(
            *(np.atleast_2d(a[index]) for a in (self.X, *self.targets)), params=dict(self.params)
        )

    def save(self, path, **options):
        """Save in any multi-array format (``.npz`` by default; also ``.h5``,
        ``.parquet``, ``.db``, ``.csv``, ...; see :func:`radar_raman.write`)."""
        from .io import save_dataset

        return save_dataset(self, path, **options)

    @classmethod
    def load(cls, path, **options) -> "SyntheticDataset":
        """Load a data set written by :meth:`save`."""
        from .io import load_dataset

        return load_dataset(path, **options)


def generate_dataset(
    n: int,
    length: int = 1024,
    min_width: int = 10,
    max_width: int = 300,
    max_num_peaks: int = 10,
    increase_noise: bool = True,
    seed: RngLike = None,
    dtype=np.float32,
    legacy_scaling: bool = False,
) -> SyntheticDataset:
    """Generate ``n`` synthetic spectra with known components.

    The defaults reproduce the original test data
    (``GenerateTestData.ipynb``: length 1024, peak widths 10-300, increased
    noise). For training-like data use ``increase_noise=False``,
    ``min_width=5, max_width=256``.

    Parameters
    ----------
    n : int
        Number of spectra.
    length : int
        Number of samples per spectrum.
    min_width, max_width : int
        Range of peak widths (in samples).
    max_num_peaks : int
        Upper bound (exclusive) on the number of bands per spectrum.
    increase_noise : bool
        Low-SNR test setting (see :func:`generate_sample`).
    seed : int, numpy Generator or None
        Random seed for reproducibility.
    dtype : numpy dtype
        Data type of the arrays.
    legacy_scaling : bool
        Reproduce the input scaling of the original training generator
        (see :func:`generate_sample`).

    Returns
    -------
    SyntheticDataset
    """
    rng = _rng(seed)
    X = np.empty((n, length), dtype=dtype)
    comps = {c: np.empty((n, length), dtype=dtype) for c in COMPONENTS}
    for i in range(n):
        s, t = generate_sample(
            length, min_width, max_width, max_num_peaks, increase_noise, rng, legacy_scaling
        )
        X[i] = s
        for c in COMPONENTS:
            comps[c][i] = t[c]
    params = dict(
        n=n,
        length=length,
        min_width=min_width,
        max_width=max_width,
        max_num_peaks=max_num_peaks,
        increase_noise=increase_noise,
        legacy_scaling=legacy_scaling,
        seed=seed if isinstance(seed, (int, type(None))) else "Generator",
    )
    return SyntheticDataset(X, **comps, params=params)


def train_test_datasets(
    n_train: int,
    n_test: int,
    length: int = 1024,
    seed: Optional[int] = 0,
    train_kwargs: Optional[dict] = None,
    test_kwargs: Optional[dict] = None,
) -> Tuple[SyntheticDataset, SyntheticDataset]:
    """Generate independent training and test sets.

    The training set uses the original training settings
    (``increase_noise=False``, widths 5-256, at most 4 bands) and the test set
    the test settings (``increase_noise=True``, widths 10-300). Override
    with ``train_kwargs`` / ``test_kwargs`` (passed to
    :func:`generate_dataset`).

    The two sets are drawn from independent random streams derived from
    ``seed``.
    """
    ss = np.random.SeedSequence(seed)
    s_train, s_test = (np.random.default_rng(s) for s in ss.spawn(2))
    tr = dict(min_width=5, max_width=256, max_num_peaks=5, increase_noise=False)
    tr.update(train_kwargs or {})
    te = dict(min_width=10, max_width=300, max_num_peaks=10, increase_noise=True)
    te.update(test_kwargs or {})
    train = generate_dataset(n_train, length, seed=s_train, **tr)
    test = generate_dataset(n_test, length, seed=s_test, **te)
    return train, test


def batch_generator(
    length: int = 750,
    batch_size: int = 100,
    min_width: int = 5,
    max_width: int = 256,
    max_num_peaks: int = 5,
    increase_noise: bool = False,
    seed: RngLike = None,
    legacy_scaling: bool = False,
) -> Iterator[Tuple[np.ndarray, Tuple[np.ndarray, ...]]]:
    """Infinite generator of training batches.

    Yields ``(X, (baseline, cosmic_rays, noise, peaks))`` with ``X`` of shape
    ``(batch_size, length, 1)`` and targets of shape ``(batch_size, length)``,
    all ``float32``. Defaults are the original training settings
    (``TrainStandard.py`` / ``TrainEnsemble.py``); ``legacy_scaling=True``
    reproduces the original generator exactly (see :func:`generate_sample`).
    """
    rng = _rng(seed)
    while True:
        X = np.empty((batch_size, length), dtype=np.float32)
        T = np.empty((4, batch_size, length), dtype=np.float32)
        for i in range(batch_size):
            s, t = generate_sample(
                length, min_width, max_width, max_num_peaks, increase_noise, rng, legacy_scaling
            )
            X[i] = s
            for j, c in enumerate(COMPONENTS):
                T[j, i] = t[c]
        yield X[..., None], tuple(T)


def make_tf_dataset(prefetch: int = 2, **kwargs):
    """Wrap :func:`batch_generator` in a ``tf.data.Dataset`` (infinite).

    Keyword arguments are passed to :func:`batch_generator`.
    """
    import tensorflow as tf

    length = kwargs.get("length", 750)
    batch_size = kwargs.get("batch_size", 100)
    spec = tf.TensorSpec((batch_size, length), tf.float32)
    ds = tf.data.Dataset.from_generator(
        lambda: batch_generator(**kwargs),
        output_signature=(tf.TensorSpec((batch_size, length, 1), tf.float32), (spec,) * 4),
    )
    return ds.prefetch(prefetch)
