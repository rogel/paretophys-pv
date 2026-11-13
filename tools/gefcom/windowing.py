"""Daily-midnight 168-to-24 window construction with leakage guards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from audit import WEATHER_COLUMNS


INCREMENT_COLUMNS = ["VAR169_INC", "VAR175_INC", "VAR178_INC", "VAR228_INC"]
KNOWN_WEATHER_COLUMNS = WEATHER_COLUMNS + INCREMENT_COLUMNS
CALENDAR_COLUMNS = ["hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos"]
MODEL_INPUT_COLUMNS = ["power_input", "target_observed"] + KNOWN_WEATHER_COLUMNS + CALENDAR_COLUMNS
SPLIT_TO_CODE = {"train": 0, "validation": 1, "test": 2}


@dataclass(frozen=True)
class StandardScalerState:
    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray
    fit_forecast_date_min: str
    fit_forecast_date_max: str
    fit_rows: int

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.columns].to_numpy(dtype=np.float64)
        return ((values - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "fit_forecast_date_min": self.fit_forecast_date_min,
            "fit_forecast_date_max": self.fit_forecast_date_max,
            "fit_rows": self.fit_rows,
        }


@dataclass
class WindowBundle:
    past_inputs: np.ndarray
    future_inputs: np.ndarray
    targets: np.ndarray
    daylight_mask: np.ndarray
    station_ids: np.ndarray
    split_codes: np.ndarray
    metadata: pd.DataFrame
    scaler: StandardScalerState


def load_spec(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_boundaries(spec: dict[str, Any]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    raw = spec["data"]["split"]["frozen_boundaries"]
    return {
        split: (pd.Timestamp(values["start"]), pd.Timestamp(values["end"]))
        for split, values in raw.items()
    }


def split_for_date(
    forecast_date: pd.Timestamp,
    boundaries: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> str | None:
    for split, (start, end) in boundaries.items():
        if start <= forecast_date <= end:
            return split
    return None


def fit_weather_scaler(
    canonical: pd.DataFrame, spec: dict[str, Any]
) -> StandardScalerState:
    boundaries = parse_boundaries(spec)
    train_start, train_end = boundaries["train"]
    train_rows = canonical[
        canonical["FORECAST_DATE"].between(train_start, train_end, inclusive="both")
    ]
    values = train_rows[KNOWN_WEATHER_COLUMNS].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return StandardScalerState(
        columns=list(KNOWN_WEATHER_COLUMNS),
        mean=mean,
        scale=scale,
        fit_forecast_date_min=str(train_rows["FORECAST_DATE"].min().date()),
        fit_forecast_date_max=str(train_rows["FORECAST_DATE"].max().date()),
        fit_rows=int(len(train_rows)),
    )


def calendar_features(timestamps: pd.Series) -> np.ndarray:
    hours = timestamps.dt.hour.to_numpy(dtype=np.float64)
    day_of_year = timestamps.dt.dayofyear.to_numpy(dtype=np.float64)
    hour_angle = 2.0 * np.pi * hours / 24.0
    year_angle = 2.0 * np.pi * (day_of_year - 1.0) / 365.25
    return np.column_stack(
        [np.sin(hour_angle), np.cos(hour_angle), np.sin(year_angle), np.cos(year_angle)]
    ).astype(np.float32)


def model_inputs(
    frame: pd.DataFrame,
    scaler: StandardScalerState,
    include_observed_power: bool,
) -> np.ndarray:
    weather = scaler.transform(frame)
    calendar = calendar_features(frame["TIMESTAMP"])
    if include_observed_power:
        power = frame["POWER"].to_numpy(dtype=np.float32)
        observed = np.ones(len(frame), dtype=np.float32)
    else:
        power = np.zeros(len(frame), dtype=np.float32)
        observed = np.zeros(len(frame), dtype=np.float32)
    return np.column_stack([power, observed, weather, calendar]).astype(np.float32)


def build_windows(canonical: pd.DataFrame, spec: dict[str, Any]) -> WindowBundle:
    history_length = int(spec["data"]["history_length"])
    prediction_length = int(spec["data"]["prediction_length"])
    if prediction_length != 24:
        raise ValueError("Daily-midnight protocol requires prediction_length=24")

    boundaries = parse_boundaries(spec)
    scaler = fit_weather_scaler(canonical, spec)
    by_zone = {
        int(zone_id): frame.sort_values("TIMESTAMP").reset_index(drop=True)
        for zone_id, frame in canonical.groupby("ZONEID", sort=True)
    }
    forecast_dates = sorted(pd.Timestamp(value) for value in canonical["FORECAST_DATE"].unique())

    past_inputs: list[np.ndarray] = []
    future_inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    daylight_masks: list[np.ndarray] = []
    station_ids: list[int] = []
    split_codes: list[int] = []
    metadata_rows: list[dict[str, Any]] = []

    for forecast_date in forecast_dates:
        split = split_for_date(forecast_date, boundaries)
        if split is None:
            continue
        issue_time = forecast_date
        history_start = issue_time - pd.Timedelta(hours=history_length - 1)
        future_end = issue_time + pd.Timedelta(hours=prediction_length)
        for zone_id, zone in by_zone.items():
            past = zone[
                zone["TIMESTAMP"].between(history_start, issue_time, inclusive="both")
            ]
            future = zone[
                zone["TIMESTAMP"].between(
                    issue_time + pd.Timedelta(hours=1), future_end, inclusive="both"
                )
            ]
            if len(past) != history_length:
                # Only the first seven forecast days lack enough context.
                continue
            if len(future) != prediction_length:
                raise ValueError(
                    f"Incomplete future window for zone={zone_id}, date={forecast_date.date()}"
                )
            expected_future = pd.date_range(
                issue_time + pd.Timedelta(hours=1), future_end, freq="h"
            )
            if not future["TIMESTAMP"].reset_index(drop=True).equals(
                pd.Series(expected_future, name="TIMESTAMP")
            ):
                raise ValueError(
                    f"Misaligned future timestamps for zone={zone_id}, date={forecast_date.date()}"
                )

            sample_id = f"z{zone_id}_{forecast_date:%Y%m%d}"
            past_inputs.append(model_inputs(past, scaler, include_observed_power=True))
            future_inputs.append(model_inputs(future, scaler, include_observed_power=False))
            targets.append(future["POWER"].to_numpy(dtype=np.float32))
            daylight_masks.append(
                (future["VAR169_INC"].to_numpy(dtype=np.float64) > 0.0).astype(np.bool_)
            )
            station_ids.append(zone_id)
            split_codes.append(SPLIT_TO_CODE[split])
            metadata_rows.append(
                {
                    "sample_id": sample_id,
                    "zone_id": zone_id,
                    "forecast_date": forecast_date,
                    "issue_time": issue_time,
                    "target_start": issue_time + pd.Timedelta(hours=1),
                    "target_end": future_end,
                    "split": split,
                }
            )

    return WindowBundle(
        past_inputs=np.stack(past_inputs),
        future_inputs=np.stack(future_inputs),
        targets=np.stack(targets),
        daylight_mask=np.stack(daylight_masks),
        station_ids=np.asarray(station_ids, dtype=np.int64),
        split_codes=np.asarray(split_codes, dtype=np.int8),
        metadata=pd.DataFrame(metadata_rows),
        scaler=scaler,
    )


def save_windows(root: Path, bundle: WindowBundle) -> None:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed / "gefcom2014_windows.npz",
        past_inputs=bundle.past_inputs,
        future_inputs=bundle.future_inputs,
        targets=bundle.targets,
        daylight_mask=bundle.daylight_mask,
        station_ids=bundle.station_ids,
        split_codes=bundle.split_codes,
    )
    bundle.metadata.to_parquet(processed / "gefcom2014_window_metadata.parquet", index=False)
    (processed / "weather_scaler.json").write_text(
        json.dumps(bundle.scaler.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "past_shape": list(bundle.past_inputs.shape),
        "future_shape": list(bundle.future_inputs.shape),
        "target_shape": list(bundle.targets.shape),
        "input_columns": MODEL_INPUT_COLUMNS,
        "known_weather_columns": KNOWN_WEATHER_COLUMNS,
        "calendar_columns": CALENDAR_COLUMNS,
        "split_codes": SPLIT_TO_CODE,
        "sample_counts": {
            key: int((bundle.metadata["split"] == key).sum()) for key in SPLIT_TO_CODE
        },
        "future_target_isolation": {
            "power_input_column": 0,
            "target_observed_column": 1,
            "future_power_placeholder": 0.0,
            "future_target_observed": 0.0,
        },
    }
    (processed / "window_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

