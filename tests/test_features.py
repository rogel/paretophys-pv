import numpy as np
import pandas as pd

from paretophys_pv.features import build_hourly_features


def test_origin_features_do_not_slide_into_target_day() -> None:
    times = pd.date_range("2006-01-01", periods=10 * 24, freq="h")
    hourly = pd.DataFrame(
        {
            "state": "tx",
            "site_key": "tx:test",
            "local_time": times,
            "split": np.where(
                times < pd.Timestamp("2006-01-08"),
                "warmup",
                np.where(times < pd.Timestamp("2006-01-10"), "train", "validation"),
            ),
            "forecast_date": times.normalize(),
            "horizon_hour": times.hour,
            "latitude": 33.0,
            "longitude": -100.0,
            "capacity_mw": 10.0,
            "actual_pu": np.arange(len(times), dtype=float) / 1000.0,
            "da_pu": 0.2,
            "solar_elevation_deg": 30.0,
            "clear_sky_ghi_wm2": 500.0,
            "daylight": True,
            "hour_sin": np.sin(2 * np.pi * times.hour / 24),
            "hour_cos": np.cos(2 * np.pi * times.hour / 24),
            "doy_sin": np.sin(2 * np.pi * times.dayofyear / 365),
            "doy_cos": np.cos(2 * np.pi * times.dayofyear / 365),
        }
    )
    features = build_hourly_features(hourly)
    day = features[features["forecast_date"] == pd.Timestamp("2006-01-08")]
    assert day["origin_trailing_actual_mean_24h_pu"].nunique() == 1
    expected = hourly.loc[6 * 24 : 7 * 24 - 1, "actual_pu"].mean()
    assert day["origin_trailing_actual_mean_24h_pu"].iloc[0] == expected
    assert day.loc[day["horizon_hour"] == 23, "previous_day_same_hour_pu"].iloc[0] == hourly.loc[6 * 24 + 23, "actual_pu"]
