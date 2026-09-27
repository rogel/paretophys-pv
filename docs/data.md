# Data sources and preparation

Download datasets from their providers and keep them under the ignored `data/` directory. Dataset access and reuse follow the providers' terms.

## NLR Solar Power Data for Integration Studies

[Official dataset page and state downloads](https://www.nlr.gov/grid/solar-power-data)

The archive contains modeled PV output and forecast series for 2006. Its `Actual` label identifies the modeled target series; it should not be interpreted as field measurements. File names encode location, PV type, capacity, year, and sampling interval.

The `nlr_data` module parses names such as `DA_32.05_-94.15_2006_UPV_95MW_60_Min.csv` and provides deterministic geographic site selection. To create canonical hourly records:

1. Match `Actual` and `DA` files by location, capacity, year, and PV type.
2. Check timestamps, missing values, duplicate intervals, and matching coverage.
3. Average each complete group of twelve 5-minute power samples into one hourly target. Retain the corresponding hourly day-ahead forecast.
4. Divide both power series by installed capacity, using consistent MW units.
5. Use local standard time, without daylight-saving shifts. Compute solar geometry at the center of each hourly interval, H:30, while retaining H:00 as its table label.
6. Assign chronological warmup, training, validation, and test periods before fitting any preprocessing statistics.

With the optional `pvlib` dependency, interval-center solar descriptors can be calculated as follows:

```python
import pandas as pd
from pvlib.location import Location
from paretophys_pv.time_alignment import hourly_interval_center_utc

labels = pd.date_range("2006-06-01", periods=24, freq="h")
utc_centers = hourly_interval_center_utc(labels, utc_offset_hours=-6)
site = Location(latitude=32.0, longitude=-100.0, tz="UTC")
position = site.get_solarposition(utc_centers)
clear_sky = site.get_clearsky(utc_centers)
```

The conversion of raw state archives to the canonical table remains dataset-specific; the `prepare` command accepts the table described below.

## GEFCom2014 Solar

- [Organizer-provided dataset information and data link](https://robjhyndman.com/publications/gefcom2014/)
- [Publisher-hosted archive](https://ars.els-cdn.com/content/image/1-s2.0-S0169207016000133-mmc1.zip)

Use the solar track within the archive. It contains normalized power and numerical weather prediction variables for three zones. Provider download endpoints may require browser access; if an older mirror is unavailable, consult the source page or publisher archive.

The raw solar-track format is not the canonical hourly input schema. A dataset-specific adapter must establish issue times, build a day-ahead reference curve from information available at issuance, and map weather descriptors into the model's feature groups. Fit any irradiance scale or normalization on training data only. Do not substitute future realized power for an unavailable forecast curve. Missing location information should be represented explicitly through a documented neutral mapping, without inventing coordinates. No automatic GEFCom conversion is bundled.

## Canonical hourly CSV

One row represents one site and hourly interval. Required columns:

| Column | Meaning |
| --- | --- |
| `state` | Geographic group used for balanced sampling and macro metrics |
| `site_key` | Globally unique site identifier |
| `local_time` | Naive local-standard-time interval start, exactly H:00 |
| `split` | `warmup`, `train`, `validation`, or `test` |
| `actual_pu` | Target power divided by site capacity |
| `da_pu` | Reference forecast available at the day's 00:00 origin |
| `latitude`, `longitude` | Site coordinates in degrees |
| `capacity_mw` | Positive installed capacity |
| `solar_elevation_deg` | Solar elevation at the hourly interval center |
| `clear_sky_ghi_wm2` | Clear-sky GHI at the hourly interval center |
| `daylight` | Boolean or 0/1 daylight indicator |

Each site needs a continuous hourly timeline and at least seven complete warmup days before its first target day. Split boundaries must occur at midnight. The splits must share a chronological order across sites. Every forecast day must have 24 hours and at least one daylight hour. The package validates these conditions instead of silently filling missing records.

`prepare` constructs calendar features, the previous day's same-hour power, seven-day same-hour means, and statistics anchored before the forecast origin. It then creates one sequence per site-day. Use only inputs genuinely available at the origin; the software cannot establish the provenance of a supplied forecast column.

## Sequence bundle

The NPZ format contains numeric arrays and Unicode metadata; loading does not enable pickle.

| Field | Shape / content |
| --- | --- |
| `history` | `(N, 168, 2)`: historical power and reference power |
| `future` | `(N, 24, 25)`: standardized features in `FEATURE_COLUMNS` order |
| `base_da`, `target` | `(N, 24)`: unstandardized per-unit values |
| `daylight` | `(N, 24)`: Boolean mask |
| `state`, `site_key`, `split`, `forecast_origin` | `(N,)`: Unicode metadata |
| `feature_mean`, `feature_std` | `(25,)`: training-only scaler |
| `feature_names` | `(25,)`: exact ordered names |

The candidate backbone consumes the 21 features belonging to S1–S4. The remaining capacity and development-region indicator columns are retained for the common input schema but excluded from the candidate model. Checkpoint prediction requires the same feature order and training scaler. For a new inference bundle, apply the saved scaler rather than refitting it.
