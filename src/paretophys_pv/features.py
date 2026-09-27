"""Leakage-safe hourly feature construction anchored at the 00:00 origin."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "da_pu",
    "da_increment_pu",
    "da_peak_pu",
    "da_peak_hour_sin",
    "da_peak_hour_cos",
    "da_energy_puh",
    "solar_proxy",
    "clear_sky_ghi_norm",
    "daylight_float",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "latitude",
    "longitude",
    "log1p_capacity_mw",
    "previous_day_same_hour_pu",
    "trailing_7_day_same_hour_mean_pu",
    "origin_trailing_actual_mean_24h_pu",
    "origin_trailing_actual_mean_72h_pu",
    "origin_trailing_abs_ramp_mean_24h_pu",
    "origin_trailing_abs_ramp_mean_72h_pu",
    "state_ny",
    "state_or",
    "state_tx",
]


def build_hourly_features(
    hourly: pd.DataFrame,
    allowed_splits: tuple[str, ...] = ("train", "validation"),
) -> pd.DataFrame:
    required = {
        "state",
        "site_key",
        "local_time",
        "split",
        "forecast_date",
        "horizon_hour",
        "latitude",
        "longitude",
        "capacity_mw",
        "actual_pu",
        "da_pu",
        "solar_elevation_deg",
        "clear_sky_ghi_wm2",
        "daylight",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    }
    missing = required - set(hourly.columns)
    if missing:
        raise ValueError(f"Missing feature-source columns: {sorted(missing)}")

    work = hourly.sort_values(["site_key", "local_time"]).copy()
    site_group = work.groupby("site_key", sort=False)
    day_group = work.groupby(["site_key", "forecast_date"], sort=False)

    work["previous_day_same_hour_pu"] = site_group["actual_pu"].shift(24)
    same_hour_lags = [site_group["actual_pu"].shift(24 * lag) for lag in range(1, 8)]
    work["trailing_7_day_same_hour_mean_pu"] = pd.concat(
        same_hour_lags, axis=1
    ).mean(axis=1)

    work["da_increment_pu"] = day_group["da_pu"].diff()
    first_hour = work["horizon_hour"] == 0
    work.loc[first_hour, "da_increment_pu"] = work.loc[first_hour, "da_pu"]
    work["da_peak_pu"] = day_group["da_pu"].transform("max")
    work["da_energy_puh"] = day_group["da_pu"].transform("sum")
    peak_rows = (
        work.loc[day_group["da_pu"].idxmax(), ["site_key", "forecast_date", "horizon_hour"]]
        .rename(columns={"horizon_hour": "da_peak_hour"})
        .drop_duplicates(["site_key", "forecast_date"])
    )
    work = work.merge(peak_rows, on=["site_key", "forecast_date"], how="left")
    work["da_peak_hour_sin"] = np.sin(
        2.0 * np.pi * work["da_peak_hour"].to_numpy(dtype=np.float64) / 24.0
    )
    work["da_peak_hour_cos"] = np.cos(
        2.0 * np.pi * work["da_peak_hour"].to_numpy(dtype=np.float64) / 24.0
    )

    # Compute rolling history at every timestamp, then retain the H:00 value
    # and broadcast it across the 24 target hours. This prevents later target
    # hours from using same-day Actual observations.
    work["absolute_ramp_pu"] = site_group["actual_pu"].diff().abs()
    ramp_group = work.groupby("site_key", sort=False)
    for hours in (24, 72):
        work[f"rolling_actual_mean_{hours}h"] = site_group["actual_pu"].transform(
            lambda series: series.shift(1).rolling(hours, min_periods=hours).mean()
        )
        work[f"rolling_abs_ramp_mean_{hours}h"] = ramp_group[
            "absolute_ramp_pu"
        ].transform(
            lambda series: series.shift(1).rolling(hours, min_periods=hours).mean()
        )
    origin = work[work["horizon_hour"] == 0][
        [
            "site_key",
            "forecast_date",
            "rolling_actual_mean_24h",
            "rolling_actual_mean_72h",
            "rolling_abs_ramp_mean_24h",
            "rolling_abs_ramp_mean_72h",
        ]
    ].rename(
        columns={
            "rolling_actual_mean_24h": "origin_trailing_actual_mean_24h_pu",
            "rolling_actual_mean_72h": "origin_trailing_actual_mean_72h_pu",
            "rolling_abs_ramp_mean_24h": "origin_trailing_abs_ramp_mean_24h_pu",
            "rolling_abs_ramp_mean_72h": "origin_trailing_abs_ramp_mean_72h_pu",
        }
    )
    work = work.merge(origin, on=["site_key", "forecast_date"], how="left")

    work["solar_proxy"] = np.maximum(
        np.sin(np.deg2rad(work["solar_elevation_deg"].to_numpy(dtype=np.float64))),
        0.0,
    )
    work["clear_sky_ghi_norm"] = work["clear_sky_ghi_wm2"] / 1000.0
    work["daylight_float"] = work["daylight"].astype(np.float64)
    work["log1p_capacity_mw"] = np.log1p(work["capacity_mw"])
    for state in ("ny", "or", "tx"):
        work[f"state_{state}"] = (work["state"] == state).astype(np.float64)

    keep = [
        "state",
        "site_key",
        "local_time",
        "split",
        "forecast_date",
        "horizon_hour",
        "actual_pu",
        "daylight",
        *FEATURE_COLUMNS,
    ]
    if not allowed_splits:
        raise ValueError("At least one feature split must be requested")
    result = work.loc[work["split"].isin(allowed_splits), keep].copy()
    if result[FEATURE_COLUMNS].isna().any().any():
        missing_counts = result[FEATURE_COLUMNS].isna().sum()
        raise ValueError(
            f"Missing engineered features: {missing_counts[missing_counts > 0].to_dict()}"
        )
    if not np.isfinite(result[FEATURE_COLUMNS].to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite engineered features")
    if set(result["split"].unique()) != set(allowed_splits):
        raise ValueError(
            f"Feature table must contain exactly the requested splits: {allowed_splits}"
        )
    return result.sort_values(["site_key", "local_time"]).reset_index(drop=True)
