# GEFCom2014 Solar preprocessing

The standalone tools in [`tools/gefcom/`](../tools/gefcom/) convert the provider's Solar Task 15 CSVs into audited hourly records, forecast windows, reference curves, and standardized feature tensors. The conversion runs on CPU and does not train forecasting models.

## Install

Use Python 3.11 or newer. From the repository root, in an activated virtual environment:

```sh
python -m pip install -r tools/gefcom/requirements.txt
python tools/gefcom/rebuild.py --help
```

PyArrow supplies Parquet support. The standalone conversion does not require PyTorch or installation of the core package.

## Provider files

Obtain the [GEFCom2014 publisher archive](https://ars.els-cdn.com/content/image/1-s2.0-S0169207016000133-mmc1.zip), following the provider's distribution terms. Extract the Solar track and Task 15 files. Retain both archives and arrange an input root as follows:

```text
input_root/
  data/raw/GEFCom2014.zip
  data/raw/source/GEFCom2014-S_V2.zip
  data/raw/source/Solar/Task 15/predictors15.csv
  data/raw/source/Solar/Task 15/train15.csv
  data/raw/source/Solar/Solution to Task 15/Solution to Task 15.csv
```

`GEFCom2014.zip` is the downloaded outer archive, renamed locally if necessary. Paths and capitalization within the extracted Solar directory must match the layout above. The tool records source hashes and cross-checks the power values in the predictor table against the training and solution files. Raw provider data are obtained separately.

## Run

From the repository root:

```sh
python tools/gefcom/rebuild.py \
  --raw-root /path/to/input_root \
  --config configs/gefcom_preprocessing.yaml \
  --output outputs/gefcom_prepared
```

Replace `/path/to/input_root` with your input directory. Use a new output directory for each run. The configuration supplies inclusive train, validation, and test forecast-origin dates. All splits must be ordered and non-overlapping. This feature mapping requires 168 history hours and a 24-hour forecast horizon.

## Processing rules

1. **Integrity and issue times.** Require the three zones, continuous hourly timestamps, unique zone/time keys, complete 24-lead days, finite values, and consistent target references. A midnight issue covers target timestamps 01:00 through the following 00:00. The last timestamp belongs to the preceding forecast-origin date. Incomplete or invalid source records fail the audit; missing values are not silently imputed.
2. **Accumulated NWP fields.** Difference VAR169, VAR175, VAR178, and VAR228 within each issue day. The first lead uses its accumulated value directly. Preserve raw increments for auditing and clip negative increments to zero for features.
3. **Windows and leakage control.** Historical power ends at the issue time. Future power and observation-indicator inputs are zero placeholders; future targets are saved separately. Fit the weather scaler using only the configured training dates.
4. **Reference and daylight.** Daylight is a positive deaccumulated VAR169 increment. Divide this increment by its training-daylight 99.5th percentile, clip to [0, 1.05], and zero outside daylight. Compute a training-only, lead-specific 99.5th-percentile envelope for the irradiance ratio, clipped to [0, 1.5].
5. **Feature mapping.** S1 uses the irradiance proxy, envelope ratio, and daylight; S2 uses the reference curve, increments, peak, peak time, and energy; S3 uses prior-day, seven-day, and trailing historical summaries; S4 uses calendar features and neutral coordinates. Unavailable coordinates and capacity use zero constants. Three zone indicators complete the 25-feature tensor. Fit its mean and standard deviation only on training samples; set constant-feature scales to one.
6. **Historical channels.** The two history channels are observed power and a binary historical NWP-daylight indicator. The latter is reconstructed from the standardized weather input. No historical reference-power curve is assumed for this dataset.

## Output interface

The output directory contains:

| Path | Contents |
| --- | --- |
| `outputs/metrics/data_audit.json` | Source schema, alignment, integrity, and target-reference checks |
| `data/interim/gefcom2014_solar_task15.parquet` | Canonical hourly records with raw and clipped increments |
| `data/processed/gefcom2014_windows.npz` | Historical/future weather windows, separate targets, zones, and split codes |
| `data/processed/weather_scaler.json` | Training-only weather scaler |
| `data/processed/gefcom/gefcom_sequences.npz` | Full mapped tensors |
| `data/processed/gefcom/gefcom_train_validation.npz` | Training/validation subset |
| `data/processed/gefcom/gefcom_test.npz` | Test subset |
| `data/processed/gefcom/gefcom_sequence_index.parquet` | Sample-to-zone/date/split mapping |
| `data/processed/gefcom/feature_mapping.json` | Exact ordered feature names, fitted scaler, reference scale, and lead envelopes |
| `data/processed/gefcom/preparation_audit.json` | Mapping, temporal isolation, and source-hash records |
| `preprocessing_config.yaml`, `preprocessing_manifest.json` | Used configuration and input/source/output hashes |

The mapped NPZ fields are `history` `(N, 168, 2)`, `future` `(N, 24, 25)`, `base_da`, `target`, and `daylight` `(N, 24)`, plus `station_ids` and `split_codes` `(N,)`. Split codes are 0 = training, 1 = validation, and 2 = test. Floating tensors use float32; masks use bool. Load NPZ files with `numpy.load(..., allow_pickle=False)`.

This dataset-specific tensor interface uses numeric split codes and a separate metadata/scaler file. It differs from the [core CLI bundle](data.md#sequence-bundle), which embeds string splits, site metadata, feature names, and scaler arrays. Do not pass these files directly to `paretophys train`; a consumer must explicitly map the documented fields and preserve the GEFCom channel meanings and fitted training scaler.

## Software checks

After installing pytest, the preprocessing-only checks can run without provider files:

```sh
python -m pytest -q tools/gefcom/tests/test_preprocessing.py
```

These checks use synthetic records to verify issue-day deaccumulation, chronological split validation, training-only scaling, and separation of future targets from inputs.
