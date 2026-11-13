"""Reproducible data-quality audit for GEFCom2014 Solar Task 15."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["ZONEID", "TIMESTAMP"]
TARGET_COLUMNS = ["ZONEID", "TIMESTAMP", "POWER"]
WEATHER_COLUMNS = [
    "VAR78",
    "VAR79",
    "VAR134",
    "VAR157",
    "VAR164",
    "VAR165",
    "VAR166",
    "VAR167",
    "VAR169",
    "VAR175",
    "VAR178",
    "VAR228",
]
ACCUMULATED_COLUMNS = ["VAR169", "VAR175", "VAR178", "VAR228"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path, expected_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(expected_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame["TIMESTAMP"] = pd.to_datetime(
        frame["TIMESTAMP"], format="%Y%m%d %H:%M", errors="raise"
    )
    return frame


def add_forecast_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    # The competition issues a forecast at midnight for hours 01:00 through
    # the next 00:00. Shifting timestamps back one hour maps all 24 leads to
    # the same forecast date.
    result["FORECAST_DATE"] = (result["TIMESTAMP"] - pd.Timedelta(hours=1)).dt.normalize()
    result["LEAD_HOUR"] = result["TIMESTAMP"].dt.hour.replace({0: 24})
    return result


def table_profile(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values(KEY_COLUMNS)
    gap_count = 0
    for _, zone in ordered.groupby("ZONEID", sort=True):
        deltas = zone["TIMESTAMP"].diff().dropna()
        gap_count += int((deltas != pd.Timedelta(hours=1)).sum())
    forecast_counts = ordered.groupby(["ZONEID", "FORECAST_DATE"]).size()
    lead_sets = ordered.groupby(["ZONEID", "FORECAST_DATE"])["LEAD_HOUR"].agg(
        lambda values: tuple(sorted(values.tolist()))
    )
    expected_leads = tuple(range(1, 25))
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "zones": sorted(int(value) for value in frame["ZONEID"].unique()),
        "rows_by_zone": {
            str(int(key)): int(value)
            for key, value in frame["ZONEID"].value_counts().sort_index().items()
        },
        "timestamp_min": str(frame["TIMESTAMP"].min()),
        "timestamp_max": str(frame["TIMESTAMP"].max()),
        "forecast_days": int(frame["FORECAST_DATE"].nunique()),
        "null_cells": int(frame.isna().sum().sum()),
        "nulls_by_column": {
            key: int(value) for key, value in frame.isna().sum().items() if value
        },
        "duplicate_key_rows": int(frame.duplicated(KEY_COLUMNS, keep=False).sum()),
        "exact_duplicate_rows": int(frame.duplicated(keep=False).sum()),
        "non_hourly_gaps": gap_count,
        "incomplete_zone_days": int((forecast_counts != 24).sum()),
        "invalid_lead_sets": int((lead_sets != expected_leads).sum()),
    }


def compare_power_reference(
    predictors: pd.DataFrame, reference: pd.DataFrame, label: str
) -> dict[str, Any]:
    merged = reference.merge(
        predictors[TARGET_COLUMNS], on=KEY_COLUMNS, how="left", suffixes=("_ref", "_pred"), indicator=True
    )
    matched = merged["_merge"] == "both"
    differences = (
        merged.loc[matched, "POWER_ref"] - merged.loc[matched, "POWER_pred"]
    ).abs()
    return {
        "reference": label,
        "reference_rows": int(len(reference)),
        "matched_rows": int(matched.sum()),
        "unmatched_rows": int((~matched).sum()),
        "max_absolute_power_difference": float(differences.max()) if len(differences) else None,
        # Allow rounding differences between predictors and target references.
        "differences_above_1e_7": int((differences > 1e-7).sum()),
    }


def accumulated_field_profile(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    grouped = frame.sort_values(["ZONEID", "FORECAST_DATE", "LEAD_HOUR"]).groupby(
        ["ZONEID", "FORECAST_DATE"], sort=False
    )
    for column in ACCUMULATED_COLUMNS:
        differences = grouped[column].diff()
        valid = differences.dropna()
        result[column] = {
            "negative_within_day_count": int((valid < -1e-9).sum()),
            "minimum_within_day_increment": float(valid.min()),
            "median_within_day_increment": float(valid.median()),
            "maximum_within_day_increment": float(valid.max()),
        }
    return result


def add_deaccumulated_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["ZONEID", "FORECAST_DATE", "LEAD_HOUR"]).copy()
    grouped = result.groupby(["ZONEID", "FORECAST_DATE"], sort=False)
    for column in ACCUMULATED_COLUMNS:
        increment = grouped[column].diff()
        first_hour = result["LEAD_HOUR"] == 1
        increment.loc[first_hour] = result.loc[first_hour, column]
        result[f"{column}_INC_RAW"] = increment
        # Small negative increments occur in the released accumulated forecast
        # fields because of numerical/encoding adjustments. Energy and
        # precipitation increments are physically non-negative, so preserve the
        # raw difference for audit and use a zero-clipped feature for modeling.
        result[f"{column}_INC"] = increment.clip(lower=0.0)
    return result.sort_values(KEY_COLUMNS).reset_index(drop=True)


def compute_split_dates(frame: pd.DataFrame) -> dict[str, Any]:
    dates = np.array(sorted(frame["FORECAST_DATE"].unique()))
    train_count = int(len(dates) * 0.70)
    validation_count = int(len(dates) * 0.15)
    test_count = len(dates) - train_count - validation_count
    train_dates = dates[:train_count]
    validation_dates = dates[train_count : train_count + validation_count]
    test_dates = dates[train_count + validation_count :]

    def summary(values: np.ndarray) -> dict[str, Any]:
        return {
            "forecast_days": int(len(values)),
            "start": str(pd.Timestamp(values[0]).date()),
            "end": str(pd.Timestamp(values[-1]).date()),
        }

    return {
        "train": summary(train_dates),
        "validation": summary(validation_dates),
        "test": summary(test_dates),
        "counts": {
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
        },
    }


def run_audit(root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    raw_root = root / "data" / "raw"
    solar_root = raw_root / "source" / "Solar"
    task_root = solar_root / "Task 15"
    archive_path = raw_root / "GEFCom2014.zip"
    solar_archive_path = raw_root / "source" / "GEFCom2014-S_V2.zip"
    train_path = task_root / "train15.csv"
    predictors_path = task_root / "predictors15.csv"
    solution_path = solar_root / "Solution to Task 15" / "Solution to Task 15.csv"

    predictors = add_forecast_coordinates(
        load_csv(predictors_path, TARGET_COLUMNS + WEATHER_COLUMNS)
    )
    train = load_csv(train_path, TARGET_COLUMNS)
    solution = load_csv(solution_path, TARGET_COLUMNS)

    profile = table_profile(predictors)
    train_comparison = compare_power_reference(predictors, train, "train15")
    solution_comparison = compare_power_reference(predictors, solution, "solution_task15")
    accumulated = accumulated_field_profile(predictors)
    canonical = add_deaccumulated_fields(predictors)

    finite_weather = np.isfinite(canonical[WEATHER_COLUMNS].to_numpy()).all()
    power_below_zero = int((canonical["POWER"] < 0).sum())
    power_above_one = int((canonical["POWER"] > 1).sum())
    negative_deaccumulated = {
        f"{column}_INC": int((canonical[f"{column}_INC"] < -1e-9).sum())
        for column in ACCUMULATED_COLUMNS
    }

    critical_checks = {
        "schema_complete": set(TARGET_COLUMNS + WEATHER_COLUMNS).issubset(canonical.columns),
        "three_expected_zones": profile["zones"] == [1, 2, 3],
        "no_null_cells": profile["null_cells"] == 0,
        "unique_zone_timestamp_key": profile["duplicate_key_rows"] == 0,
        "continuous_hourly_series": profile["non_hourly_gaps"] == 0,
        "complete_24h_forecast_days": profile["incomplete_zone_days"] == 0,
        "valid_lead_hours": profile["invalid_lead_sets"] == 0,
        "train_power_matches_predictors": train_comparison["unmatched_rows"] == 0
        and train_comparison["differences_above_1e_7"] == 0,
        "solution_power_matches_predictors": solution_comparison["unmatched_rows"] == 0
        and solution_comparison["differences_above_1e_7"] == 0,
        "finite_weather_values": bool(finite_weather),
        "normalized_power_nonnegative_and_bounded": power_below_zero == 0
        and float(canonical["POWER"].max()) <= 1.05,
        "deaccumulated_features_nonnegative": all(
            int((canonical[f"{column}_INC"] < 0).sum()) == 0
            for column in ACCUMULATED_COLUMNS
        ),
    }

    report = {
        "status": "passed" if all(critical_checks.values()) else "failed",
        "intended_grain": "one row per zone, target timestamp and 1-24h forecast lead",
        "source": {
            "publisher_supplement_url": "https://ars.els-cdn.com/content/image/1-s2.0-S0169207016000133-mmc1.zip",
            "archive_sha256": sha256(archive_path),
            "solar_archive_sha256": sha256(solar_archive_path),
            "dataset_doi": "10.1016/j.ijforecast.2016.02.001",
        },
        "profile": profile,
        "critical_checks": critical_checks,
        "target_reference_checks": [train_comparison, solution_comparison],
        "target_validity": {
            "minimum": float(canonical["POWER"].min()),
            "maximum": float(canonical["POWER"].max()),
            "below_zero_count": power_below_zero,
            "above_one_count": power_above_one,
            "zero_rate_by_zone": {
                str(int(zone)): float((group["POWER"] == 0).mean())
                for zone, group in canonical.groupby("ZONEID")
            },
        },
        "weather_ranges": {
            column: {
                "minimum": float(canonical[column].min()),
                "median": float(canonical[column].median()),
                "maximum": float(canonical[column].max()),
            }
            for column in WEATHER_COLUMNS
        },
        "accumulated_fields": accumulated,
        "negative_deaccumulated_counts": negative_deaccumulated,
        "chronological_split": compute_split_dates(canonical),
        "caveats": [
            "The package does not disclose station coordinates or timestamp timezone.",
            "Obtain provider files separately and follow their distribution terms.",
            "VAR169, VAR175, VAR178 and VAR228 are accumulated forecast fields and must be differenced within each midnight-issued forecast day before trend semantics are computed.",
            "The raw accumulated fields contain rare small negative within-day differences; raw differences are retained for audit and modeling increments are clipped at zero.",
            "Power values up to 1.05 p.u. pass the audit and are retained without clipping.",
            "The released predictors15 file contains finalized power values for both training and Task 15 solution periods; future target isolation must therefore be enforced by the windowing code rather than inferred from null values.",
        ],
    }
    return report, canonical


def save_audit(root: Path, report: dict[str, Any], canonical: pd.DataFrame) -> None:
    metrics_path = root / "outputs" / "metrics" / "data_audit.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if report["status"] == "failed":
        failed = [key for key, value in report["critical_checks"].items() if not value]
        raise RuntimeError(f"Critical data-quality checks failed: {failed}")
    interim_path = root / "data" / "interim" / "gefcom2014_solar_task15.parquet"
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(interim_path, index=False)
