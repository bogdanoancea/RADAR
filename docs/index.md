# RADAR

**R**aman spectral **A**nalysis using **D**eep learning for **A**rtifact **R**emoval.

RADAR decomposes a Raman spectrum into four additive components:

| Component | Attribute | Typical origin |
|---|---|---|
| Baseline | `baseline` | fluorescence, substrate, optics |
| Cosmic rays | `cosmic_rays` | high-energy particles hitting the detector (1–2 pixel spikes) |
| Noise | `noise` | shot / read-out noise |
| Peaks | `peaks` | Raman bands: the corrected, denoised spectrum |

The decomposition is done by one of two convolutional networks, each trained
on 240 million synthetic spectra with known components:

- **`ensemble`** (default): a two-stage network with 0.28 M parameters.
- **`standard`**: a single-stage network with 0.59 M parameters.

```python
import radar_raman
comps = radar_raman.decompose("my_spectra.spc")  # or a NumPy array, or any other supported file
comps.peaks                                 # corrected spectra
comps.save("my_spectra_components.h5")
```

## Where to go next

- **[User guide](user_guide.md)**: the complete guide (installation,
  formats, decomposition, results, models, training, fine-tuning, synthetic
  data, evaluation, command line, troubleshooting)
- [Models](models.md): the architectures and the pre-trained weights
- [Synthetic data](data_generation.md)
- [Command line](cli.md)
- [API reference](api.md)
- [Differences from the original code](differences.md)
- [Changelog](changelog.md)

## How to cite

If you use RADAR, please cite the software:

> Sjöberg, J., & Oancea, B. (2026). *RADAR: Raman spectral Analysis using Deep
> learning for Artifact Removal (Python package)* (Version 0.1.0) [Computer
> software]. Zenodo. <https://doi.org/10.5281/zenodo.23760379>

`radar_raman.citation()` prints the references, and
`CITATION.cff` in the repository has them in machine-readable form.

## Credits

The networks, the data generator and the pre-trained weights come from the
original implementation by Joel Sjöberg. The Python package is by Bogdan
Oancea. Licence: BSD 3-Clause.
