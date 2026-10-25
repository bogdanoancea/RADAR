# Authors and credits

## Original author

**Joel Sjöberg** wrote the original implementation on which RADAR is based:
the two network architectures (standard and ensemble), the synthetic
spectrum generator, the training scripts and the pre-trained weights shipped
in `src/radar/weights/`.

- Original repository: <https://github.com/JoelSjoberg/Comprehensive-Raman-Spectral-Analysis-Using-Deep-Learning>
- Pre-trained weights: `Models/standardModel.h5` → `weights/standard.h5` and
  `Models/ensembleModelRes.h5` → `weights/ensemble.h5`. The files are copied
  unchanged (SHA-256 checksums are listed in `docs/models.md`).

## Python package

**Bogdan Oancea** turned the original scripts into this package: the module
structure, the port to Keras 3, the legacy weight loader, the simple API and
the `Decomposer` class, the file I/O (including the native SPC and JCAMP-DX
readers and writers), the command-line interface, metrics, tests, examples and
documentation.

## Third-party notes

- A comment in the original `Scripts/essentials.py` points to the ORPL package
  (<https://github.com/mr-sheg/orpl>) for its "bubble fill" baseline
  algorithm, which the original author used as a comparison method. RADAR
  contains no ORPL code.
- The SPC and JCAMP-DX readers are implemented from the public format
  specifications and contain no third-party code.
- Optional readers that RADAR imports only if they are installed:
  `renishawWiRE` (MIT) for Renishaw `.wdf`, and `rosettasciio` (GPL-3.0) for
  Horiba LabSpec `.xml` and TriVista `.tvf`. Neither is distributed with
  RADAR.
