#!/usr/bin/env python3
"""Build GEFCom reference curves and feature tensors from audited windows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_mapping(source: Path, output_dir: Path) -> None:
    paths = {key: source / relative for key, relative in {
        "windows": "data/processed/gefcom2014_windows.npz",
        "metadata": "data/processed/gefcom2014_window_metadata.parquet",
        "weather_scaler": "data/processed/weather_scaler.json",
        "window_manifest": "data/processed/window_manifest.json",
        "source_audit": "outputs/metrics/data_audit.json",
    }.items()}
    source_hashes = {key: sha256(path) for key, path in paths.items()}
    source_audit = json.loads(paths["source_audit"].read_text())
    if not all(source_audit["critical_checks"].values()):
        raise RuntimeError("Source GEFCom integrity audit did not pass")
    arrays = np.load(paths["windows"])
    metadata = pd.read_parquet(paths["metadata"])
    scaler = json.loads(paths["weather_scaler"].read_text())
    split_codes = arrays["split_codes"].astype(np.int8)
    train = split_codes == 0
    future_known = arrays["future_inputs"][:, :, 2:].astype(np.float64)
    raw_weather = future_known[:, :, :16] * np.asarray(scaler["scale"])[None, None, :] + np.asarray(scaler["mean"])[None, None, :]
    raw_irradiance = np.maximum(raw_weather[:, :, 12], 0.0)
    daylight = arrays["daylight_mask"].astype(bool)
    train_positive = raw_irradiance[train][daylight[train]]
    irradiance_scale = float(np.quantile(train_positive, 0.995))
    curve = np.clip(raw_irradiance / irradiance_scale, 0.0, 1.05)
    curve[~daylight] = 0.0
    lead_envelope = np.quantile(raw_irradiance[train], 0.995, axis=0)
    lead_envelope = np.maximum(lead_envelope, irradiance_scale * 1e-6)
    clear_proxy = np.clip(raw_irradiance / lead_envelope[None, :], 0.0, 1.5)
    clear_proxy[~daylight] = 0.0

    sample_count = len(split_codes)
    future = np.zeros((sample_count, 24, 25), dtype=np.float64)
    future[:, :, 0] = curve
    future[:, 0, 1] = curve[:, 0]
    future[:, 1:, 1] = np.diff(curve, axis=1)
    peak = curve.max(axis=1)
    peak_hour = curve.argmax(axis=1)
    future[:, :, 2] = peak[:, None]
    future[:, :, 3] = np.sin(2.0 * np.pi * peak_hour[:, None] / 24.0)
    future[:, :, 4] = np.cos(2.0 * np.pi * peak_hour[:, None] / 24.0)
    future[:, :, 5] = curve.sum(axis=1)[:, None]
    future[:, :, 6] = curve
    future[:, :, 7] = clear_proxy
    future[:, :, 8] = daylight.astype(np.float64)
    future[:, :, 9:13] = future_known[:, :, 16:20]
    # Dataset coordinates and capacities are undisclosed. Neutral constants are
    # used; they become zero after the train-only standardization safeguard.
    future[:, :, 13:16] = 0.0

    past_power = arrays["past_inputs"][:, :, 0].astype(np.float64)
    previous_day = past_power[:, -24:]
    trailing_7 = past_power.reshape(sample_count, 7, 24).mean(axis=1)
    future[:, :, 16] = previous_day
    future[:, :, 17] = trailing_7
    for hours, power_index, ramp_index in ((24, 18, 20), (72, 19, 21)):
        history = past_power[:, -hours:]
        future[:, :, power_index] = history.mean(axis=1)[:, None]
        future[:, :, ramp_index] = np.abs(np.diff(history, axis=1)).mean(axis=1)[:, None]
    zones = arrays["station_ids"].astype(np.int64)
    for zone in (1, 2, 3):
        future[:, :, 21 + zone] = (zones == zone)[:, None]

    feature_mean = future[train].reshape(-1, 25).mean(axis=0)
    feature_std = future[train].reshape(-1, 25).std(axis=0)
    constant = feature_std < 1e-8
    feature_std[constant] = 1.0
    future_standardized = ((future - feature_mean[None, None, :]) / feature_std[None, None, :]).astype(np.float32)

    past_full = arrays["past_inputs"].astype(np.float64)
    past_raw_irradiance = past_full[:, :, 14] * float(scaler["scale"][12]) + float(scaler["mean"][12])
    history = np.stack([past_power, past_raw_irradiance > 0.0], axis=-1).astype(np.float32)
    target = arrays["targets"].astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "gefcom_sequences.npz", history=history, future=future_standardized,
                        base_da=curve.astype(np.float32), target=target, daylight=daylight,
                        station_ids=zones, split_codes=split_codes)
    for name, selected in (("train_validation", split_codes != 2), ("test", split_codes == 2)):
        np.savez_compressed(
            output_dir / f"gefcom_{name}.npz",
            history=history[selected], future=future_standardized[selected],
            base_da=curve[selected].astype(np.float32), target=target[selected],
            daylight=daylight[selected], station_ids=zones[selected],
            split_codes=split_codes[selected],
        )
    metadata = metadata.copy()
    metadata.insert(0, "sample_index", np.arange(len(metadata), dtype=np.int64))
    metadata.to_parquet(output_dir / "gefcom_sequence_index.parquet", index=False)

    split_counts = {name: int((metadata["split"] == name).sum()) for name in ("train", "validation", "test")}
    chronological = bool((metadata["target_start"] > metadata["issue_time"]).all())
    no_future_target_input = bool(np.all(arrays["future_inputs"][:, :, :2] == 0.0))
    audit = {
        "status": "passed" if chronological and no_future_target_input else "failed",
        "source_sha256": source_hashes,
        "source_critical_checks": source_audit["critical_checks"],
        "rows": int(len(metadata)), "zones": sorted(np.unique(zones).tolist()),
        "split_counts": split_counts,
        "shapes": {"history": list(history.shape), "future": list(future_standardized.shape), "target": list(target.shape)},
        "target_starts_after_issue": chronological,
        "future_source_power_and_observed_placeholders_zero": no_future_target_input,
        "feature_scaler_fit_split": "train_only",
        "irradiance_scale_train_daylight_q995": irradiance_scale,
        "constant_neutral_feature_indices": np.flatnonzero(constant).tolist(),
        "test_labels_used_for_feature_construction": False,
        "training_artifact_excludes_test": True,
    }
    write_json(output_dir / "preparation_audit.json", audit)
    write_json(output_dir / "feature_mapping.json", {
        "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
        "leadwise_irradiance_envelope": lead_envelope.tolist(),
        "irradiance_scale": irradiance_scale,
        "feature_names": [
            "nwp_curve_pu", "nwp_increment_pu", "nwp_peak_pu", "nwp_peak_hour_sin", "nwp_peak_hour_cos", "nwp_energy_puh",
            "solar_proxy", "leadwise_envelope_ratio", "daylight_float", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
            "neutral_latitude", "neutral_longitude", "neutral_log_capacity", "previous_day_same_hour_pu", "trailing_7_day_same_hour_mean_pu",
            "origin_trailing_actual_mean_24h_pu", "origin_trailing_actual_mean_72h_pu", "origin_trailing_abs_ramp_mean_24h_pu",
            "origin_trailing_abs_ramp_mean_72h_pu", "zone_1", "zone_2", "zone_3"
        ],
    })
    if audit["status"] != "passed":
        raise RuntimeError("GEFCom preparation audit failed")
