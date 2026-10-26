# Command-line interface

Installing the package adds a `radar-raman` command, also available as
`python -m radar_raman.cli`. `radar <command> --help` shows every option.

## `radar-raman decompose`

```bash
radar-raman decompose INPUT [INPUT ...] -o OUTPUT [-m ensemble|standard|FILE] [--architecture A]
                [--units original|normalized] [--no-normalize] [--batch-size 32]
                [read options] [--out-layout wide|long]
```

This decomposes spectra from one or more files of any readable format (see
`radar-raman formats`). The output format follows the extension of `-o`. With
several inputs, `-o` is either a folder (results are named
`<stem>_components.h5`) or a pattern containing `{stem}`.

```bash
radar-raman decompose sample.wdf -o sample_components.h5
radar-raman decompose spectrum.txt -o spectrum_components.csv -m standard
radar-raman decompose data/*.spc -o "results/{stem}_components.parquet"
radar-raman decompose maps.h5 --key data -o out.npz -m my_model.keras
radar-raman decompose export.csv --decimal , -o out.db
```

## `radar-raman convert`

```bash
radar-raman convert INPUT OUTPUT [read options] [--out-layout wide|long]
```

This converts spectra, or a decomposition written by RADAR, between any two
formats:

```bash
radar-raman convert measurement.spc measurement.jdx
radar-raman convert map.wdf map.parquet
radar-raman convert result.h5 result.xlsx
```

## Read options

These apply to `decompose` and `convert`:

| Option | Meaning |
|---|---|
| `--format NAME` | input format when the extension is ambiguous (`csv`, `spc`, `jcamp`, ...) |
| `--layout auto\|wide\|columns\|rows\|long` | table layout of text / Excel / Parquet / SQLite input |
| `--key NAME` | array name in `.npz`, `.h5`, `.mat`, `.json` |
| `--sheet NAME` | Excel sheet |
| `--table NAME` / `--query SQL` | SQLite table, or any SQL query returning the spectra |
| `--delimiter C` / `--decimal .\|,` | text field separator / decimal mark |
| `--out-layout wide\|long` | table layout of the output |

## `radar-raman generate`

```bash
radar-raman generate -n N -o OUTPUT [--length 1024] [--train] [--min-width W] [--max-width W] [--seed 0]
```

This writes a synthetic data set with ground truth. By default it uses the
test settings; `--train` switches to the training settings. The output can be
any multi-array format (`.npz`, `.h5`, `.parquet`, `.db`, `.csv`, ...).

## `radar-raman train`

```bash
radar-raman train -o MODEL.keras [-a ensemble|standard] [--init pretrained|FILE] [--data FILE]
            [--epochs 10] [--steps 1000] [--batch-size 100] [--length 750]
            [--learning-rate LR] [--validation-size 1000] [--checkpoint BEST.keras] [--seed 0]
```

This trains a new model, or fine-tunes one with `--init`. Without `--data`
it trains on freshly generated synthetic batches. Without `--learning-rate`
it uses the original 12-level schedule; pass a small value such as `1e-5`
when fine-tuning.

```bash
# full schedule of the pre-trained models: 240 M spectra (use a GPU)
radar-raman train -a ensemble --epochs 12 --steps 200000 -o ensemble_retrained.keras
# fine-tune the pre-trained standard model on your own synthetic set
radar-raman generate -n 5000 --length 1340 --train -o my_train.npz
radar-raman train -a standard --init pretrained --data my_train.npz --epochs 5 --learning-rate 1e-5 -o tuned.keras
```

## `radar-raman evaluate`

```bash
radar-raman evaluate DATASET [-m ensemble|standard|FILE] [--statistics]
```

This prints RMSE, MAE, max error and R² for each component as JSON. With
`--statistics` it prints the mean, median and variance over spectra of the
per-spectrum RMSE and max error instead.

## `radar-raman formats`, `radar-raman info`

`radar-raman formats` lists the supported file formats and their requirements.

`radar-raman info` prints the package, TensorFlow and Keras versions, where the
pre-trained weights are, and which optional vendor readers are installed.

`radar-raman info MODEL` prints what a saved model was made with: the RADAR, Keras,
TensorFlow, Python and NumPy versions, the architecture, the training
settings, and whether the installed versions can load it.

```bash
radar-raman info my_model.keras
radar-raman info ensemble
```
