# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Joel Sjöberg (original implementation)
# Copyright (c) 2026 Bogdan Oancea (RADAR package)
"""File formats: read, decompose and write spectra in many formats.

Run:  python examples/08_file_formats.py

Shows
  * the list of supported formats,
  * reading any file with radar.read (format from the extension),
  * a Raman map: decomposition, map images, a band image,
  * writing results to CSV, Excel, Parquet, SQLite, HDF5, SPC, JCAMP-DX, ...,
  * converting between formats.
"""
from pathlib import Path

import numpy as np

import radar

HERE = Path(__file__).parent
OUT = HERE / "output" / "formats"
OUT.mkdir(parents=True, exist_ok=True)

# 1. What can be read and written?
radar.formats(show=True)

# 2. Read the example CSV (wavenumber column + one column per spectrum)
data = radar.read(HERE / "data" / "synthetic_spectra.csv")
print("\nread:", data)

# 3. Write the same spectra in several formats and read them back
for ext in ["csv", "txt", "json", "h5", "npz", "mat", "db", "spc", "jdx", "xlsx", "parquet"]:
    try:
        path = radar.write(data, OUT / f"spectra.{ext}")
        back = radar.read(path)
        same = np.allclose(back.intensities, data.intensities, rtol=1e-5)
        print(f"  .{ext:8s} -> {Path(path).name:22s} read back identical: {same}")
    except ImportError as e:  # Excel / Parquet need pandas + openpyxl / pyarrow
        print(f"  .{ext:8s} skipped ({e})")

# 4. A small Raman map (3 x 4 points) built from synthetic spectra
cube = radar.generate(12, length=1024, seed=5).X.reshape(3, 4, 1024) * 3000 + 400
wn = np.linspace(400, 1800, 1024)
comps = radar.decompose(cube, wavenumbers=wn)           # map shape is kept
print("\nmap decomposition:", comps)
peaks_map = comps.as_map("peaks")                       # (3, 4, 1024)
band = comps.band_image(990, 1010)                      # integrated band, (3, 4)
print("peaks map:", peaks_map.shape, " band image:\n", np.round(band, 1))

# 5. Save the decomposition: one file (multi-array formats) or one file per
#    component (single-array formats such as SPC and JCAMP-DX)
for name in ["map_components.h5", "map_components.csv", "map_components.db", "map_components.spc"]:
    out = comps.save(OUT / name)
    print("saved", out if not isinstance(out, list) else [p.name for p in out])
again = radar.load_components(OUT / "map_components.db")
print("reloaded from SQLite:", again)

# 6. Convert between formats (same as: radar convert in.spc out.jdx)
radar.write(radar.read(OUT / "spectra.spc"), OUT / "converted.jdx")
print("converted SPC -> JCAMP-DX:", radar.read(OUT / "converted.jdx"))

# 7. SQL query on a SQLite file: just spectrum 2
d2 = radar.read(OUT / "spectra.db", query="SELECT * FROM spectra WHERE spectrum = 2")
print("SQL query result:", d2)
