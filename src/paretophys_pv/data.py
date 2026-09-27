"""Portable hourly-table and sequence input, with training-only feature scaling."""

from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, build_hourly_features

ARRAY_NAMES = ("history", "future", "base_da", "target", "daylight")
META_NAMES = ("state", "site_key", "split", "forecast_origin")


def validate_bundle(data: dict[str, np.ndarray]) -> None:
    required = {*ARRAY_NAMES, *META_NAMES, "feature_mean", "feature_std", "feature_names"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Missing sequence fields: {sorted(missing)}")
    n = len(data["target"])
    if not n:
        raise ValueError("Sequence bundle is empty")
    shapes = {"history": (n, 168, 2), "future": (n, 24, len(FEATURE_COLUMNS)),
              "base_da": (n, 24), "target": (n, 24), "daylight": (n, 24)}
    for name, shape in shapes.items():
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError(f"Invalid shape or non-finite values in {name}; expected {shape}")
    if data["daylight"].dtype != np.bool_ or not data["daylight"].any(axis=1).all():
        raise ValueError("Every forecast needs a Boolean mask with at least one daylight hour")
    for name in META_NAMES:
        if data[name].shape != (n,):
            raise ValueError(f"Invalid metadata shape: {name}")
    if list(data["feature_names"]) != FEATURE_COLUMNS:
        raise ValueError("Feature order does not match FEATURE_COLUMNS")
    for name in ("feature_mean", "feature_std"):
        if data[name].shape != (len(FEATURE_COLUMNS),) or not np.isfinite(data[name]).all():
            raise ValueError(f"Invalid scaler: {name}")
    if np.any(data["feature_std"] <= 0):
        raise ValueError("Feature standard deviations must be positive")
    index = pd.DataFrame({name: data[name] for name in META_NAMES})
    index["forecast_origin"] = pd.to_datetime(index["forecast_origin"], errors="raise")
    if index.duplicated(["site_key", "forecast_origin"]).any():
        raise ValueError("Duplicate site-day forecasts")
    if not set(index["split"]).issubset({"train", "validation", "test"}):
        raise ValueError("Sequence splits must be train, validation, or test")
    present = [name for name in ("train", "validation", "test") if name in set(index["split"])]
    for earlier, later in zip(present, present[1:]):
        if index.loc[index.split == earlier, "forecast_origin"].max() >= index.loc[index.split == later, "forecast_origin"].min():
            raise ValueError("Splits must follow a common chronological ordering across sites")


def load_bundle(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    validate_bundle(data)
    return data


def save_bundle(path: str | Path, data: dict[str, np.ndarray]) -> None:
    validate_bundle(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def prepare_hourly(hourly: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convert complete local-standard-time hourly records to 168-to-24 sequences."""
    hourly = hourly.copy()
    required = {"state", "site_key", "local_time", "split", "actual_pu", "da_pu",
                "latitude", "longitude", "capacity_mw", "solar_elevation_deg",
                "clear_sky_ghi_wm2", "daylight"}
    if required - set(hourly.columns):
        raise ValueError(f"Missing hourly columns: {sorted(required - set(hourly.columns))}")
    hourly["local_time"] = pd.to_datetime(hourly["local_time"], errors="raise")
    times = hourly["local_time"].dt
    if times.tz is not None or (times.minute != 0).any() or (times.second != 0).any():
        raise ValueError("Use naive local-standard-time labels at exact H:00")
    if not set(hourly["split"]).issubset({"warmup", "train", "validation", "test"}):
        raise ValueError("Unknown split label")
    if hourly["daylight"].isna().any() or not hourly["daylight"].isin([True, False, 0, 1]).all():
        raise ValueError("daylight must contain Boolean or 0/1 values")
    hourly["daylight"] = hourly["daylight"].astype(bool)
    numeric = ["actual_pu", "da_pu", "latitude", "longitude", "capacity_mw",
               "solar_elevation_deg", "clear_sky_ghi_wm2"]
    if not np.isfinite(hourly[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Hourly inputs contain non-finite values")
    if (hourly["capacity_mw"] <= 0).any() or (hourly[["actual_pu", "da_pu"]] < 0).any().any():
        raise ValueError("Capacity must be positive and normalized power nonnegative")
    hourly = hourly.sort_values(["site_key", "local_time"]).reset_index(drop=True)
    if hourly.duplicated(["site_key", "local_time"]).any():
        raise ValueError("Duplicate site-hour records")
    for site, group in hourly.groupby("site_key"):
        if group["state"].nunique() != 1 or not group.local_time.diff().dropna().eq(pd.Timedelta(hours=1)).all():
            raise ValueError(f"Site {site} has a changing group or a discontinuous hourly timeline")
    hourly["forecast_date"] = hourly.local_time.dt.normalize()
    hourly["horizon_hour"] = hourly.local_time.dt.hour
    for key, values in (("hour", hourly.horizon_hour / 24), ("doy", hourly.local_time.dt.dayofyear / 365)):
        hourly[f"{key}_sin"] = np.sin(2 * np.pi * values)
        hourly[f"{key}_cos"] = np.cos(2 * np.pi * values)
    splits = tuple(name for name in ("train", "validation", "test") if name in set(hourly["split"]))
    if "train" not in splits or "validation" not in splits:
        raise ValueError("Preparing a bundle requires train and validation periods")
    features = build_hourly_features(hourly, allowed_splits=splits).set_index(["site_key", "local_time"])
    values = {name: [] for name in (*ARRAY_NAMES, *META_NAMES)}
    for site_key, site in hourly.groupby("site_key", sort=True):
        site = site.reset_index(drop=True)
        for pos in site.index[(site.horizon_hour == 0) & site["split"].isin(splits)]:
            past, day = site.iloc[max(0, pos - 168):pos], site.iloc[pos:pos + 24]
            if len(past) != 168 or len(day) != 24 or day["split"].nunique() != 1:
                raise ValueError(f"Incomplete history/horizon or split within a day at {site_key}, row {pos}")
            ix = pd.MultiIndex.from_arrays([[site_key] * 24, day.local_time])
            values["history"].append(past[["actual_pu", "da_pu"]].to_numpy(np.float32))
            values["future"].append(features.loc[ix, FEATURE_COLUMNS].to_numpy(np.float32))
            for name, col in [("base_da", "da_pu"), ("target", "actual_pu"), ("daylight", "daylight")]:
                values[name].append(day[col].to_numpy())
            for name in ("state", "site_key", "split"):
                values[name].append(str(day[name].iloc[0]))
            values["forecast_origin"].append(day.local_time.iloc[0].isoformat())
    data = {name: np.asarray(value, dtype=str if name in META_NAMES else bool if name == "daylight" else np.float32) for name, value in values.items()}
    train = data["future"][data["split"] == "train"].reshape(-1, len(FEATURE_COLUMNS))
    data["feature_mean"] = train.mean(axis=0)
    data["feature_std"] = train.std(axis=0)
    data["feature_std"][data["feature_std"] < 1e-6] = 1
    data["future"] = (data["future"] - data["feature_mean"]) / data["feature_std"]
    data["feature_names"] = np.asarray(FEATURE_COLUMNS)
    validate_bundle(data)
    return data


def demo_hourly(seed: int = 2026) -> pd.DataFrame:
    """Generate toy records locally for software checks, not performance evaluation."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-06-01", periods=30 * 24, freq="h")
    day_number = np.arange(len(times)) // 24
    sun = np.maximum(np.sin(np.pi * (times.hour.to_numpy() - 6) / 12), 0)
    daylight = (times.hour >= 7) & (times.hour <= 17)
    rows = []
    for state, latitude, longitude in [("tx", 32., -100.), ("ny", 42., -75.)]:
        base = sun * np.repeat(rng.uniform(.55, .9, 30), 24)
        actual = np.clip(base * 1.04 + rng.normal(0, .025, len(times)), 0, 1.05)
        base[~daylight] = 0; actual[~daylight] = 0
        rows.append(pd.DataFrame({"state": state, "site_key": f"{state}:demo", "local_time": times,
            "split": np.select([day_number < 7, day_number < 19, day_number < 25], ["warmup", "train", "validation"], default="test"),
            "actual_pu": actual, "da_pu": base, "latitude": latitude, "longitude": longitude,
            "capacity_mw": 10., "solar_elevation_deg": np.rad2deg(np.arcsin(sun)),
            "clear_sky_ghi_wm2": 900 * sun, "daylight": daylight}))
    return pd.concat(rows, ignore_index=True)
