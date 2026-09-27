"""Group-balanced daytime and tail-error metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _top_fraction_mean(values: np.ndarray, fraction: float = 0.10) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    count = max(1, int(math.ceil(values.size * fraction)))
    return float(np.sort(values)[-count:].mean())


def evaluate_predictions(
    frame: pd.DataFrame,
    prediction_col: str,
    *,
    clip_lower: float = 0.0,
    clip_upper: float = 1.05,
    expected_split: str = "validation",
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate point forecasts with explicit split selection.

    Daytime nMAE is capacity-normalized MAE because targets are already in p.u.
    The two primary objectives are macro averages over states, preventing states
    with fewer integrity-eligible sites from receiving lower weight.
    """

    required = {
        "state",
        "site_key",
        "local_time",
        "split",
        "forecast_date",
        "horizon_hour",
        "actual_pu",
        "daylight",
        prediction_col,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")
    if set(frame["split"].unique()) != {expected_split}:
        if expected_split == "validation":
            raise ValueError("Metrics require validation rows only")
        raise ValueError(f"Metrics require {expected_split} rows only")
    if frame.duplicated(["site_key", "local_time"]).any():
        raise ValueError("Duplicate site-hour rows in prediction frame")

    work = frame.copy()
    prediction = pd.to_numeric(work[prediction_col], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(prediction).all():
        raise ValueError(f"Non-finite predictions in {prediction_col}")
    prediction = np.clip(prediction, clip_lower, clip_upper)
    prediction[~work["daylight"].to_numpy(dtype=bool)] = 0.0
    work["prediction_pu"] = prediction
    work["absolute_error_pu"] = np.abs(
        prediction - work["actual_pu"].to_numpy(dtype=np.float64)
    )
    work["squared_error_pu"] = (
        prediction - work["actual_pu"].to_numpy(dtype=np.float64)
    ) ** 2

    daytime = work[work["daylight"]].copy()
    site_day = (
        daytime.groupby(["state", "site_key", "forecast_date"], as_index=False)
        .agg(
            daytime_nmae_pu=("absolute_error_pu", "mean"),
            daytime_nrmse_pu=("squared_error_pu", lambda x: float(np.sqrt(x.mean()))),
            daylight_hours=("daylight", "size"),
        )
        .sort_values(["state", "site_key", "forecast_date"])
    )

    ramp_rows: list[dict] = []
    energy_rows: list[dict] = []
    for (state, site_key, forecast_date), day in work.groupby(
        ["state", "site_key", "forecast_date"], sort=True
    ):
        day = day.sort_values("horizon_hour")
        actual = day["actual_pu"].to_numpy(dtype=np.float64)
        pred = day["prediction_pu"].to_numpy(dtype=np.float64)
        ramp_rows.append(
            {
                "state": state,
                "site_key": site_key,
                "forecast_date": forecast_date,
                "ramp_mae_pu": float(np.mean(np.abs(np.diff(pred) - np.diff(actual)))),
            }
        )
        energy_rows.append(
            {
                "state": state,
                "site_key": site_key,
                "forecast_date": forecast_date,
                "daily_energy_abs_bias_puh": float(abs(pred.sum() - actual.sum())),
            }
        )
    ramp = pd.DataFrame(ramp_rows)
    energy = pd.DataFrame(energy_rows)

    per_state_rows: list[dict] = []
    for state in sorted(work["state"].unique()):
        state_daytime = daytime[daytime["state"] == state]
        state_site_day = site_day[site_day["state"] == state]
        per_state_rows.append(
            {
                "state": state,
                "site_count": int(work.loc[work["state"] == state, "site_key"].nunique()),
                "forecast_days": int(
                    work.loc[work["state"] == state, "forecast_date"].nunique()
                ),
                "daytime_nmae_percent": float(
                    state_daytime["absolute_error_pu"].mean() * 100.0
                ),
                "daytime_nrmse_percent": float(
                    np.sqrt(state_daytime["squared_error_pu"].mean()) * 100.0
                ),
                "site_day_cvar90_percent": float(
                    _top_fraction_mean(
                        state_site_day["daytime_nmae_pu"].to_numpy()
                    )
                    * 100.0
                ),
                "ramp_mae_percent": float(
                    ramp.loc[ramp["state"] == state, "ramp_mae_pu"].mean()
                    * 100.0
                ),
                "daily_energy_abs_bias_puh": float(
                    energy.loc[
                        energy["state"] == state, "daily_energy_abs_bias_puh"
                    ].mean()
                ),
            }
        )
    per_state = pd.DataFrame(per_state_rows)

    horizon = (
        daytime.groupby("horizon_hour", as_index=False)
        .agg(daytime_mae_pu=("absolute_error_pu", "mean"), rows=("actual_pu", "size"))
        .sort_values("horizon_hour")
    )
    horizon["daytime_mae_percent"] = horizon["daytime_mae_pu"] * 100.0

    summary = {
        "prediction_column": prediction_col,
        "split": expected_split,
        "states": sorted(work["state"].unique().tolist()),
        "site_count": int(work["site_key"].nunique()),
        "forecast_days": int(work["forecast_date"].nunique()),
        "rows": int(len(work)),
        "daytime_rows": int(len(daytime)),
        "state_macro_daytime_nmae_percent": float(
            per_state["daytime_nmae_percent"].mean()
        ),
        "state_macro_daytime_nrmse_percent": float(
            per_state["daytime_nrmse_percent"].mean()
        ),
        "state_macro_site_day_cvar90_percent": float(
            per_state["site_day_cvar90_percent"].mean()
        ),
        "state_macro_ramp_mae_percent": float(per_state["ramp_mae_percent"].mean()),
        "state_macro_daily_energy_abs_bias_puh": float(
            per_state["daily_energy_abs_bias_puh"].mean()
        ),
        "pooled_daytime_nmae_percent_secondary": float(
            daytime["absolute_error_pu"].mean() * 100.0
        ),
    }
    return summary, per_state, site_day, horizon
