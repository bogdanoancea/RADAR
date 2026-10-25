# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""High-level interface: :class:`Decomposer`.

Typical use::

    import radar

    model = radar.load_pretrained("ensemble")   # or radar.load_model("my.keras")
    comps = model.decompose(spectra)            # spectra: (n, length) or (length,)
    comps.baseline, comps.cosmic_rays, comps.noise, comps.peaks
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Sequence, Union

import keras
import numpy as np

from . import _weights
from .components import Components
from .models import ARCHITECTURES, COMPONENTS, build_model, compile_model
from .preprocessing import Scaling, denormalize_components, minmax_normalize

__all__ = ["Decomposer", "load_model", "load_pretrained", "new_model", "decompose", "train", "evaluate",
           "model_info"]


def _infer_architecture(model: keras.Model) -> str:
    return _architecture_from_name(model.name or "")


def _architecture_from_name(name: str) -> str:
    for arch in ARCHITECTURES:
        if name == f"radar_{arch}" or name.startswith(f"radar_{arch}"):
            return arch
    return "custom"


class Decomposer:
    """A RADAR network wrapped with pre/post-processing, training and I/O.

    Create one with :meth:`pretrained`, :meth:`new` or :meth:`load` (or the
    module-level shortcuts :func:`radar.load_pretrained` and
    :func:`radar.load_model`).

    Parameters
    ----------
    model : keras.Model
        A network that maps ``(batch, length, 1)`` to
        ``[baseline, cosmic_rays, noise, peaks]``.
    architecture : str, optional
        ``"standard"``, ``"ensemble"`` or ``"custom"``. Inferred from the
        model name when omitted.

    Attributes
    ----------
    model : keras.Model
        The underlying Keras model.
    architecture : str
    history : keras.callbacks.History or None
        History of the last :meth:`train` call.
    """

    def __init__(self, model: keras.Model, architecture: Optional[str] = None):
        self.model = model
        self.architecture = architecture or _infer_architecture(model)
        self.history = None
        #: where the weights come from (shown by :meth:`info`)
        self.origin = ""
        #: settings of the training that produced the weights, if known
        self.training_info: Optional[dict] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def new(cls, architecture: str = "ensemble", learning_rate=1e-5, **build_kwargs) -> "Decomposer":
        """A new, untrained (randomly initialised) network.

        Parameters
        ----------
        architecture : {"ensemble", "standard"}
        learning_rate : float or schedule
            Initial optimiser setting (``train`` re-compiles anyway).
        **build_kwargs
            Hyper-parameters passed to the builder, e.g. ``scaler``,
            ``dim_red_size``, ``seed`` (see :mod:`radar.models`).
        """
        model = build_model(architecture, **build_kwargs)
        compile_model(model, architecture, learning_rate)
        dec = cls(model, architecture)
        dec.origin = "new network (random initialisation)"
        return dec

    @classmethod
    def pretrained(cls, architecture: str = "ensemble") -> "Decomposer":
        """The network with the pre-trained weights of the original
        implementation (J. Sjöberg), trained on 240 million synthetic spectra.

        Parameters
        ----------
        architecture : {"ensemble", "standard"}
            ``"ensemble"`` (two-stage model, 0.28 M parameters) or
            ``"standard"`` (single-stage model, 0.59 M parameters).
        """
        dec = cls.new(architecture)
        _weights.load_legacy_weights(
            dec.model, _weights.pretrained_weights_path(architecture), architecture
        )
        dec.origin = f"bundled pre-trained weights ({_weights.PRETRAINED_FILES[architecture]})"
        dec.training_info = dict(_weights.PRETRAINED_TRAINING)
        return dec

    @classmethod
    def load(cls, path, architecture: Optional[str] = None, compile: bool = True) -> "Decomposer":
        """Load a model saved with :meth:`save`, or a weight file.

        Supported files
        ---------------
        ``*.keras``
            Complete model (architecture + weights + optimiser state).
        ``*.weights.h5``
            Keras 3 weights; ``architecture`` is taken from the file's RADAR
            record when present, otherwise it is required.
        ``*.h5`` written by the original scripts (``model.save_weights``)
            Keras 2 weights, e.g. ``Models/ensembleModelRes.h5``;
            ``architecture`` is required.

        Notes
        -----
        A ``.keras`` file written by a newer Keras version may not load in an
        older one (Keras adds new fields to the stored configuration). For
        RADAR architectures the file is then loaded by rebuilding the network
        from its stored hyper-parameters and reading only its weights; the
        optimiser state is not restored in that case (a warning is issued).

        Files saved by RADAR record the software versions they were saved
        with (see :func:`radar.model_info`); a warning names the versions to
        install when the file comes from newer software than the installed one.
        """
        from . import _provenance

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        info = _provenance.read_info(path)
        for note in _provenance.compatibility(info):
            if "upgrade" in note:  # saved with newer software than installed
                warnings.warn(f"{path.name}: {note}", UserWarning, stacklevel=3)
        dec = cls._load_file(path, architecture, compile, info)
        dec.origin = info.get("origin") or f"file {path.name}"
        if info.get("training"):
            dec.training_info = info["training"]
        dec.saved_info = info
        return dec

    @classmethod
    def _load_file(cls, path: Path, architecture, compile, info) -> "Decomposer":
        if path.suffix == ".keras":
            # importing radar registers the custom layers/losses
            try:
                with warnings.catch_warnings():
                    # an optimizer that was never used has no state to restore
                    warnings.filterwarnings("ignore", message=".*Skipping variable loading for optimizer.*")
                    model = keras.models.load_model(path, compile=compile)
            except Exception as err:  # e.g. saved with a newer Keras version
                dec = cls._load_keras_weights_only(path, err)
                return dec
            return cls(model, architecture)
        if path.suffix in (".h5", ".hdf5"):
            hyper = {}
            if architecture is None and info.get("architecture") in ARCHITECTURES:
                architecture = info["architecture"]
                hyper = dict(info.get("hyperparameters") or {})
            if architecture not in ARCHITECTURES:
                raise ValueError(
                    "Loading a weight file requires architecture='standard' or 'ensemble'"
                )
            dec = cls.new(architecture, **hyper)
            if _weights.is_legacy_weight_file(path):
                _weights.load_legacy_weights(dec.model, path, architecture)
            else:
                dec.model.load_weights(path)
            return dec
        raise ValueError(f"Unsupported model file {path} (use .keras, .weights.h5 or .h5)")

    @classmethod
    def _load_keras_weights_only(cls, path: Path, original_error: Exception) -> "Decomposer":
        """Fallback loader: rebuild a RADAR architecture and read the weights
        stored inside a ``.keras`` archive."""
        import json
        import tempfile
        import zipfile

        try:
            with zipfile.ZipFile(path) as zf:
                config = json.loads(zf.read("config.json"))
                cfg = config["config"]
                architecture = _architecture_from_name(cfg.get("name", ""))
                if architecture not in ARCHITECTURES:
                    raise original_error
                layers = {l["config"]["name"]: l["config"] for l in cfg["layers"]}
                reg = layers["enc_conv1"].get("kernel_regularizer") or {}
                reg = reg.get("config", {})
                hyper = dict(
                    scaler=int(layers["enc_conv1"]["filters"]),
                    dim_red_size=int(layers["key_conv"]["filters"]),
                    l1_param=float(reg.get("l1", 1e-6)),
                    l2_param=float(reg.get("l2", 1e-6)),
                )
                with tempfile.TemporaryDirectory() as tmp:
                    wpath = Path(tmp) / "model.weights.h5"
                    wpath.write_bytes(zf.read("model.weights.h5"))
                    dec = cls.new(architecture, **hyper)
                    dec.model.load_weights(wpath)
        except Exception:
            raise original_error
        reason = str(original_error).strip().splitlines()[-1] if str(original_error).strip() else ""
        warnings.warn(
            f"{path.name} could not be loaded directly by Keras {keras.__version__} "
            f"({reason}); it was probably saved with a "
            "newer Keras version. The network was rebuilt and its weights loaded; the optimiser "
            "state was not restored.",
            RuntimeWarning,
            stacklevel=3,
        )
        return dec

    def save(self, path, overwrite: bool = True) -> Path:
        """Save the model.

        * ``*.keras`` -- complete model (recommended); reload with
          :meth:`load` / :func:`radar.load_model` without extra arguments.
        * ``*.weights.h5`` -- weights only; reload with
          :func:`radar.load_model` (the architecture is stored in the file).

        Both files also record the RADAR, Keras, TensorFlow, Python and NumPy
        versions, the architecture, its hyper-parameters and the training
        settings (see :func:`radar.model_info`).
        """
        from . import _provenance

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        name = path.name
        if name.endswith(".weights.h5"):
            self.model.save_weights(path, overwrite=overwrite)
        elif path.suffix == ".keras":
            self.model.save(path, overwrite=overwrite)
        else:
            raise ValueError("File name must end with '.keras' or '.weights.h5'")
        _provenance.write_info(path, _provenance.make_info(self))
        return path

    def info(self) -> dict:
        """Provenance of this model: architecture, hyper-parameters, origin,
        training settings, and the versions of the software it runs on now."""
        from . import _provenance

        rec = _provenance.make_info(self)
        rec.pop("date_saved", None)
        rec["running_with"] = rec.pop("saved_with")
        saved = getattr(self, "saved_info", None)
        if saved:
            rec["saved_with"] = saved.get("saved_with", {})
            if saved.get("date_saved"):
                rec["date_saved"] = saved["date_saved"]
        return rec

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_normalized(self, spectra: np.ndarray, batch_size: int = 32, verbose: int = 0) -> dict:
        """Run the network on spectra that are already scaled to [0, 1].

        Parameters
        ----------
        spectra : ndarray, shape (n, length)

        Returns
        -------
        dict
            ``{component: ndarray (n, length)}`` in the normalised scale.
        """
        x = np.asarray(spectra, dtype="float32")[..., None]
        if x.ndim != 3 or len(x) == 0:
            raise ValueError(f"expected a non-empty (n, length) array, got shape {x.shape[:-1]}")
        parts = [[] for _ in COMPONENTS]
        n_batches = int(np.ceil(len(x) / batch_size))
        for b in range(n_batches):
            # direct (eager) call: no re-tracing when spectrum lengths change
            out = self.model(x[b * batch_size:(b + 1) * batch_size], training=False)
            for j, o in enumerate(out):
                parts[j].append(keras.ops.convert_to_numpy(o))
            if verbose:
                print(f"\rdecomposed batch {b + 1}/{n_batches}", end="", flush=True)
        if verbose:
            print()
        return {c: np.concatenate(p).astype(np.float64) for c, p in zip(COMPONENTS, parts)}

    def decompose(
        self,
        spectra,
        wavenumbers: Optional[np.ndarray] = None,
        normalize: bool = True,
        units: str = "original",
        batch_size: int = 32,
        verbose: int = 0,
        output=None,
        **read_options,
    ) -> Union[Components, List[Components]]:
        """Split Raman spectra into baseline, cosmic rays, noise and peaks.

        Parameters
        ----------
        spectra : array-like, SpectralData, or file path
            * 1-D array ``(length,)`` -- one spectrum;
            * 2-D array ``(n, length)`` -- ``n`` spectra of equal length;
            * 3-D array ``(ny, nx, length)`` -- a map (``map_shape`` is kept);
            * list of 1-D arrays -- spectra of possibly different lengths;
            * :class:`radar.io.SpectralData` (from :func:`radar.read`), whose
              wavenumbers, names, positions and map shape are carried over;
            * a file path in any format readable by :func:`radar.read`
              (``read_options`` are passed to it).
        wavenumbers : ndarray, optional
            Spectral axis, stored in the result (not used by the network).
        normalize : bool
            Min-max normalise each spectrum to [0, 1] before prediction
            (default, as the networks were trained on [0, 1] spectra). Use
            ``False`` only for spectra that are already in that range.
        units : {"original", "normalized"}
            Units of the returned components. ``"original"`` maps them back to
            the input intensities (the components then sum to the input);
            ``"normalized"`` returns the network output as is.
        batch_size : int
            Prediction batch size.
        verbose : int
            Print batch progress if non-zero.
        output : str or Path, optional
            Also save the result to this file (any format writable by
            :func:`radar.write`).

        Returns
        -------
        Components
            For array input (1-D or 2-D, or a list of equal-length spectra).
        list of Components
            For a list of spectra of different lengths (one per spectrum).

        Notes
        -----
        The networks accept any spectrum length. They were trained on spectra
        of 750 points with band widths of 5-256 points; spectra whose bands are
        much wider or narrower (in points) may be decomposed less accurately.
        """
        if units not in ("original", "normalized"):
            raise ValueError("units must be 'original' or 'normalized'")
        result = self._decompose(spectra, wavenumbers, normalize, units, batch_size, verbose,
                                 read_options)
        if output is not None:
            if isinstance(result, list):
                raise ValueError("spectra of different lengths: save each Components separately")
            result.save(output)
        return result

    def _decompose(self, spectra, wavenumbers, normalize, units, batch_size, verbose, read_options):
        from pathlib import Path as _Path

        info = {}
        if isinstance(spectra, (str, _Path)):
            from .io import read

            spectra = read(spectra, **read_options)
        elif read_options:
            raise TypeError(f"unexpected arguments {sorted(read_options)}")

        # SpectralData from radar.io
        if hasattr(spectra, "intensities"):
            if wavenumbers is None:
                wavenumbers = getattr(spectra, "wavenumbers", None)
            info = dict(names=spectra.names, positions=spectra.positions,
                        position_names=list(spectra.position_names), map_shape=spectra.map_shape,
                        x_units=spectra.x_units)
            spectra = spectra.intensities

        if isinstance(spectra, (list, tuple)):
            arrays = [np.asarray(s, dtype=np.float64).ravel() for s in spectra]
            lengths = {len(a) for a in arrays}
            if len(lengths) > 1:
                wns = wavenumbers if isinstance(wavenumbers, list) else None
                return self._decompose_ragged(arrays, normalize, units, batch_size, verbose, wns,
                                              info.get("names"))
            spectra = np.stack(arrays)
            if isinstance(wavenumbers, list):
                wavenumbers = wavenumbers[0]

        arr = np.asarray(spectra, dtype=np.float64)
        if arr.ndim > 2:  # a map (ny, nx, length)
            info.setdefault("map_shape", tuple(arr.shape[:-1]))
            info["map_shape"] = info["map_shape"] or tuple(arr.shape[:-1])
            arr = arr.reshape(-1, arr.shape[-1])
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"spectra must be 1-D or 2-D, got shape {arr.shape}")
        if arr.shape[1] < 2:
            raise ValueError("spectra must contain at least 2 points")
        if not np.all(np.isfinite(arr)):
            raise ValueError("spectra contain NaN or infinite values")
        if wavenumbers is not None and len(wavenumbers) != arr.shape[1]:
            raise ValueError(
                f"wavenumbers has length {len(wavenumbers)} but spectra have {arr.shape[1]} points"
            )

        if normalize:
            x, scaling = minmax_normalize(arr)
        else:
            x, scaling = arr, Scaling.identity(len(arr))

        comps = self.predict_normalized(x, batch_size=batch_size, verbose=verbose)
        if units == "original":
            comps = denormalize_components(comps, scaling)
            inp = arr
        else:
            inp = x

        if single:
            inp = inp[0]
            comps = {k: v[0] for k, v in comps.items()}
        return Components(
            spectra=inp,
            **comps,
            wavenumbers=None if wavenumbers is None else np.asarray(wavenumbers),
            units=units if normalize else "original",
            model=self.architecture,
            meta={"offset": scaling.offset, "scale": scaling.scale, "normalized_input": normalize},
            **info,
        )

    def _decompose_ragged(self, arrays, normalize, units, batch_size, verbose, wavenumbers=None,
                          names=None):
        groups = defaultdict(list)
        for i, a in enumerate(arrays):
            groups[len(a)].append(i)
        result: List[Optional[Components]] = [None] * len(arrays)
        for _, idx in groups.items():
            comps = self.decompose(
                np.stack([arrays[i] for i in idx]),
                normalize=normalize,
                units=units,
                batch_size=batch_size,
                verbose=verbose,
            )
            for j, i in enumerate(idx):
                c = comps[j]
                c.meta = {k: (v[j] if isinstance(v, np.ndarray) else v) for k, v in comps.meta.items()}
                if wavenumbers is not None:
                    c.wavenumbers = np.asarray(wavenumbers[i])
                if names is not None:
                    c.names = [names[i]]
                result[i] = c
        return result

    __call__ = decompose

    # ------------------------------------------------------------------
    # Training and evaluation
    # ------------------------------------------------------------------
    def train(
        self,
        data=None,
        *,
        epochs: int = 10,
        steps_per_epoch: int = 1000,
        batch_size: int = 100,
        learning_rate: Union[str, float, keras.optimizers.schedules.LearningRateSchedule] = "original",
        length: int = 750,
        min_width: int = 5,
        max_width: int = 256,
        max_num_peaks: int = 5,
        increase_noise: bool = False,
        legacy_scaling: bool = False,
        validation_data="synthetic",
        validation_size: int = 1000,
        validation_length: int = 2000,
        seed: Optional[int] = 0,
        callbacks: Optional[Sequence[keras.callbacks.Callback]] = None,
        checkpoint_path: Optional[str] = None,
        verbose: int = 1,
    ):
        """Train (or fine-tune) the network.

        Parameters
        ----------
        data : None, SyntheticDataset or (X, targets)
            ``None`` (default): train on an endless stream of freshly
            generated synthetic spectra, ``steps_per_epoch`` batches per epoch.
            Otherwise a fixed data set: a
            :class:`~radar.generator.SyntheticDataset` or a tuple ``(X, y)``
            with ``X`` of shape ``(n, length)`` and ``y`` a list
            ``[baseline, cosmic_rays, noise, peaks]`` (or a dict with these
            keys) of arrays ``(n, length)``; ``steps_per_epoch`` is then ignored.
        epochs, steps_per_epoch, batch_size : int
            The bundled pre-trained models were trained for 12 epochs x
            200 000 steps of 100 spectra, i.e. 240 million spectra.
        learning_rate : "original", float or Keras schedule
            ``"original"``: the 12-level piecewise-constant decay used to train
            the pre-trained models, stretched to ``epochs * steps_per_epoch``
            steps (see
            :func:`radar.training.original_learning_rate_schedule`); it starts at
            1.05e-3, so for *fine-tuning* a pre-trained model pass a small
            constant such as ``1e-5`` instead.
        length, min_width, max_width, max_num_peaks, increase_noise, legacy_scaling
            Settings of the synthetic training spectra (used when
            ``data is None``); see :func:`radar.generator.generate_sample`.
        validation_data : "synthetic", None, SyntheticDataset or (X, y)
            ``"synthetic"`` generates ``validation_size`` spectra of
            ``validation_length`` points as in the original training scripts.
        seed : int or None
            Seed of the synthetic data streams.
        callbacks : list of keras callbacks, optional
        checkpoint_path : str, optional
            Save the best model (lowest validation loss) to this ``.keras`` file.
        verbose : int
            Keras verbosity.

        Returns
        -------
        keras.callbacks.History
        """
        from .training import fit, original_learning_rate_schedule

        lr_spec = learning_rate if isinstance(learning_rate, (str, int, float)) else type(learning_rate).__name__

        if isinstance(learning_rate, str):
            if learning_rate != "original":
                raise ValueError("learning_rate must be 'original', a float or a schedule")
            if data is None:
                total = epochs * steps_per_epoch
            else:
                n = len(data) if hasattr(data, "X") else len(data[0])
                total = epochs * int(np.ceil(n / batch_size))
            learning_rate = original_learning_rate_schedule(total)

        if self.architecture in ARCHITECTURES:
            compile_model(self.model, self.architecture, learning_rate)
        else:
            self.model.optimizer.learning_rate = learning_rate  # custom model: keep its compile

        self.history = fit(
            self.model,
            data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            length=length,
            min_width=min_width,
            max_width=max_width,
            max_num_peaks=max_num_peaks,
            increase_noise=increase_noise,
            legacy_scaling=legacy_scaling,
            validation_data=validation_data,
            validation_size=validation_size,
            validation_length=validation_length,
            seed=seed,
            callbacks=callbacks,
            checkpoint_path=checkpoint_path,
            verbose=verbose,
        )
        if data is None:
            n_seen = epochs * steps_per_epoch * batch_size
            data_desc = f"synthetic stream (length={length}, widths {min_width}-{max_width}, " \
                        f"max_num_peaks={max_num_peaks}, increase_noise={increase_noise}, " \
                        f"legacy_scaling={legacy_scaling}, seed={seed})"
        else:
            n = len(data) if hasattr(data, "X") else len(data[0])
            n_seen = epochs * n
            data_desc = f"fixed data set of {n} spectra"
        record = {
            "epochs": epochs, "steps_per_epoch": steps_per_epoch if data is None else None,
            "batch_size": batch_size, "learning_rate": lr_spec, "spectra_seen": int(n_seen),
            "data": data_desc,
            "final_loss": float(self.history.history["loss"][-1]) if self.history.history.get("loss") else None,
        }
        if self.training_info:  # fine-tuning: keep what came before
            record["started_from"] = self.training_info
        self.training_info = record
        return self.history

    def evaluate(self, dataset, normalize: bool = False, batch_size: int = 32,
                 statistics: bool = False) -> dict:
        """Compare the decomposition of a data set with its ground truth.

        Parameters
        ----------
        dataset : SyntheticDataset
            E.g. from :func:`radar.generate_dataset`.
        normalize : bool
            Whether to min-max normalise the inputs first. Synthetic data are
            already in the network's scale, so the default is ``False``.
        statistics : bool
            If ``True``, return the mean, median and variance over spectra of
            the per-spectrum RMSE and max error instead.

        Returns
        -------
        dict
            ``{component: {"rmse", "mae", "max_error", "r2"}}`` (see
            :func:`radar.metrics.evaluate_components`), or with
            ``statistics=True`` ``{component: {"rmse_mean", "rmse_median",
            "rmse_variance", "max_error_mean", "max_error_median",
            "max_error_variance"}}`` (see :func:`radar.metrics.error_statistics`).
        """
        from .metrics import error_statistics, evaluate_components

        comps = self.decompose(dataset.X, normalize=normalize, units="original", batch_size=batch_size)
        if statistics:
            return error_statistics(dataset, comps)
        return evaluate_components(dataset, comps)

    # ------------------------------------------------------------------
    def summary(self, **kwargs):
        """Print the Keras model summary."""
        return self.model.summary(**kwargs)

    @property
    def n_parameters(self) -> int:
        return int(self.model.count_params())

    def __repr__(self) -> str:
        return f"Decomposer(architecture={self.architecture!r}, parameters={self.n_parameters:,})"


# ----------------------------------------------------------------------
# Module-level API (re-exported as radar.load_model, radar.decompose, ...)
# ----------------------------------------------------------------------
_PRETRAINED_CACHE: dict = {}


def load_model(model="ensemble", architecture: Optional[str] = None, compile: bool = True) -> Decomposer:
    """Get a model: a pre-trained one by name, or one saved in a file.

    Parameters
    ----------
    model : str, Path or Decomposer
        ``"ensemble"`` (default) or ``"standard"`` for the bundled pre-trained
        networks; otherwise a path to a ``.keras`` file, a ``.weights.h5``
        file or a Keras 2 ``.h5`` weight file. A :class:`Decomposer` is
        returned unchanged.
    architecture : {"ensemble", "standard"}, optional
        Required for weight-only files.
    compile : bool
        Compile the model when loading a ``.keras`` file.
    """
    if isinstance(model, Decomposer):
        return model
    from pathlib import Path as _Path

    if isinstance(model, str) and model in ARCHITECTURES and not _Path(model).exists():
        return Decomposer.pretrained(model)
    return Decomposer.load(model, architecture=architecture, compile=compile)


def load_pretrained(architecture: str = "ensemble") -> Decomposer:
    """The bundled pre-trained network (same as ``load_model("ensemble")``)."""
    return Decomposer.pretrained(architecture)


def new_model(architecture: str = "ensemble", **hyper_parameters) -> Decomposer:
    """A new, untrained network (see :meth:`Decomposer.new`)."""
    return Decomposer.new(architecture, **hyper_parameters)


def _cached(model) -> Decomposer:
    if isinstance(model, Decomposer):
        return model
    if isinstance(model, str) and model in ARCHITECTURES:
        if model not in _PRETRAINED_CACHE:
            _PRETRAINED_CACHE[model] = Decomposer.pretrained(model)
        return _PRETRAINED_CACHE[model]
    return load_model(model)


def decompose(spectra, model="ensemble", output=None, **kwargs):
    """Decompose spectra in one call.

    Parameters
    ----------
    spectra : array, SpectralData or file path
        See :meth:`Decomposer.decompose`; a file in any supported format is
        read first.
    model : "ensemble", "standard", path to a saved model, or Decomposer
        Pre-trained models are loaded once and cached.
    output : str or Path, optional
        Save the result to this file (format from the extension).
    **kwargs
        Passed to :meth:`Decomposer.decompose` (``normalize``, ``units``,
        ``wavenumbers``, ``batch_size``, file-reading options, ...).

    Examples
    --------
    >>> comps = radar.decompose("map.wdf", output="map_components.h5")   # doctest: +SKIP
    """
    return _cached(model).decompose(spectra, output=output, **kwargs)


def train(data=None, *, model="ensemble", init: Optional[str] = None, save_to=None, **train_options):
    """Train a new network, or fine-tune an existing one, in one call.

    Parameters
    ----------
    data : None, SyntheticDataset, (X, y), or path
        Training data; ``None`` trains on freshly generated synthetic spectra.
        A path is read with :func:`radar.load_dataset`.
    model : "ensemble", "standard" or Decomposer
        Architecture of a new network, or an existing model to continue
        training.
    init : None, "pretrained", or path
        Start from random weights (``None``), from the bundled pre-trained
        weights, or from a saved model.
    save_to : str or Path, optional
        Save the trained model here (``.keras`` or ``.weights.h5``).
    **train_options
        Passed to :meth:`Decomposer.train` (``epochs``, ``steps_per_epoch``,
        ``batch_size``, ``learning_rate``, ``validation_data``, ...).

    Returns
    -------
    Decomposer
        The trained model (its Keras history is in ``.history``).
    """
    if isinstance(model, Decomposer):
        dec = model
    elif init is None:
        dec = Decomposer.new(model)
    elif init == "pretrained":
        dec = Decomposer.pretrained(model)
    else:
        dec = Decomposer.load(init, architecture=model)
    if isinstance(data, (str,)) or hasattr(data, "__fspath__"):
        from .io import load_dataset

        data = load_dataset(data)
    dec.train(data, **train_options)
    if save_to is not None:
        dec.save(save_to)
    return dec


def evaluate(model, dataset, statistics: bool = False, **kwargs) -> dict:
    """Error metrics of ``model`` on a data set with known components.

    ``model`` may be a name, a path or a :class:`Decomposer`; ``dataset`` a
    :class:`~radar.generator.SyntheticDataset` or a path to one. See
    :meth:`Decomposer.evaluate`.
    """
    if isinstance(dataset, str) or hasattr(dataset, "__fspath__"):
        from .io import load_dataset

        dataset = load_dataset(dataset)
    return _cached(model).evaluate(dataset, statistics=statistics, **kwargs)


def model_info(model="ensemble", show: bool = False) -> dict:
    """Provenance of a model, without loading it when ``model`` is a file.

    Parameters
    ----------
    model : str, Path or Decomposer
        ``"ensemble"`` / ``"standard"`` (bundled weights), a saved model file
        (``.keras``, ``.weights.h5``, Keras 2 ``.h5``), or a loaded model.
    show : bool
        Also print a readable summary, including any version incompatibility
        with the installed software.

    Returns
    -------
    dict
        ``saved_with`` (RADAR, Keras, TensorFlow, Python, NumPy, h5py versions
        and platform), ``architecture``, ``hyperparameters``,
        ``n_parameters``, ``origin``, ``date_saved``, ``training`` (settings
        and number of spectra seen) and ``compatibility`` (notes; empty when
        the installed versions can load the file natively).
    """
    from pathlib import Path as _Path

    from . import _provenance

    if isinstance(model, Decomposer):
        info = model.info()
    elif isinstance(model, str) and model in ARCHITECTURES and not _Path(model).exists():
        info = Decomposer.pretrained(model).info()  # bundled: always loadable
    else:
        info = _provenance.read_info(model)
    info["compatibility"] = _provenance.compatibility(info) if "saved_with" in info else []
    if "running_with" in info:  # a loaded model: it already runs here
        info["compatibility"] = []
    if show:
        print(_provenance.describe(info))
        for note in info["compatibility"]:
            print("compatibility:", note)
        if not info["compatibility"]:
            print("compatibility: loads with the installed versions")
    return info
