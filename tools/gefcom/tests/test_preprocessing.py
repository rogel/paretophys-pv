"""Synthetic checks for the standalone GEFCom data conversion."""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import ACCUMULATED_COLUMNS, add_deaccumulated_fields, add_forecast_coordinates
from rebuild import validate_spec
from windowing import KNOWN_WEATHER_COLUMNS, build_windows, fit_weather_scaler


def spec():
    return {"format_version": 1, "data": {
        "history_length": 168, "prediction_length": 24,
        "split": {"frozen_boundaries": {
            "train": {"start": "2012-04-01", "end": "2012-04-08"},
            "validation": {"start": "2012-04-09", "end": "2012-04-10"},
            "test": {"start": "2012-04-11", "end": "2012-04-12"},
        }},
    }}


def canonical():
    times = pd.date_range("2012-04-01 01:00", periods=12 * 24, freq="h")
    frame = pd.DataFrame({"ZONEID": 1, "TIMESTAMP": times, "POWER": 0.5})
    frame = add_forecast_coordinates(frame)
    for column in KNOWN_WEATHER_COLUMNS:
        frame[column] = np.arange(len(frame), dtype=float) + 1
    return frame


def test_midnight_belongs_to_previous_issue_day():
    frame = add_forecast_coordinates(pd.DataFrame({
        "TIMESTAMP": pd.to_datetime(["2012-04-02 00:00", "2012-04-02 01:00"])
    }))
    assert frame["LEAD_HOUR"].tolist() == [24, 1]
    assert frame["FORECAST_DATE"].dt.strftime("%Y-%m-%d").tolist() == ["2012-04-01", "2012-04-02"]


def test_deaccumulation_resets_daily_and_preserves_negative_raw_increment():
    frame = add_forecast_coordinates(pd.DataFrame({
        "ZONEID": [1] * 4,
        "TIMESTAMP": pd.to_datetime(["2012-04-01 01:00", "2012-04-01 02:00",
                                     "2012-04-02 01:00", "2012-04-02 02:00"]),
    }))
    for column in ACCUMULATED_COLUMNS:
        frame[column] = [3.0, 8.0, 2.0, 1.5]
    result = add_deaccumulated_fields(frame)
    for column in ACCUMULATED_COLUMNS:
        assert result[f"{column}_INC_RAW"].tolist() == [3.0, 5.0, 2.0, -0.5]
        assert result[f"{column}_INC"].tolist() == [3.0, 5.0, 2.0, 0.0]


def test_scaler_ignores_validation_and_test_weather():
    frame = canonical()
    before = fit_weather_scaler(frame, spec())
    later = frame["FORECAST_DATE"] > pd.Timestamp("2012-04-08")
    frame.loc[later, KNOWN_WEATHER_COLUMNS] = 1e9
    after = fit_weather_scaler(frame, spec())
    np.testing.assert_array_equal(before.mean, after.mean)
    np.testing.assert_array_equal(before.scale, after.scale)
    assert before.fit_rows == 8 * 24


def test_future_targets_do_not_enter_window_inputs():
    frame = canonical()
    before = build_windows(frame, spec())
    last = frame["FORECAST_DATE"] == pd.Timestamp("2012-04-12")
    frame.loc[last, "POWER"] = 999.0
    after = build_windows(frame, spec())
    np.testing.assert_array_equal(before.past_inputs, after.past_inputs)
    np.testing.assert_array_equal(before.future_inputs, after.future_inputs)
    assert np.all(after.future_inputs[:, :, :2] == 0)
    assert np.all(after.targets[-1] == 999.0)
    assert (after.metadata["target_start"] > after.metadata["issue_time"]).all()
    assert after.split_codes.tolist() == [0, 1, 1, 2, 2]


@pytest.mark.parametrize("invalid", ["overlap", "history"])
def test_invalid_window_configuration_is_rejected(invalid):
    config = deepcopy(spec())
    if invalid == "overlap":
        config["data"]["split"]["frozen_boundaries"]["validation"]["start"] = "2012-04-08"
    else:
        config["data"]["history_length"] = 24
    with pytest.raises(ValueError):
        validate_spec(config)
