# Example data

**These are synthetic spectra, not measurements.** They were generated with
`radar_raman.generate_dataset(5, length=1024, seed=42)` (the default test-data
settings) so that the examples run without downloading anything.

- `synthetic_spectra.csv` — 5 spectra, one per column, with a Raman-shift
  axis from 400 to 1800 cm⁻¹ in the first column. Intensities were rescaled
  to detector-count-like units: `4000 * X + 600`.
- `synthetic_spectra_truth.npz` — the same 5 spectra in the normalised scale
  (`X`) with their true components (`baseline`, `cosmic_rays`, `noise`,
  `peaks`); load with `radar_raman.load_dataset(...)`.

Real spectra from the original project (live cells and glioma tissue) are
available at <https://seafile.utu.fi/d/dd325af0161b41bc989c/>.
