import pandas as pd
import pytest

from paretophys_pv.metrics import evaluate_predictions


def toy_frame() -> pd.DataFrame:
    rows = []
    for state in ("a", "b"):
        for hour, actual, prediction, daylight in (
            (0, 0.0, 0.2, False),
            (1, 0.5, 0.4, True),
            (2, 1.0, 0.8, True),
        ):
            rows.append(
                {
                    "state": state,
                    "site_key": f"{state}:site",
                    "local_time": pd.Timestamp("2006-09-01")
                    + pd.Timedelta(hours=hour),
                    "split": "validation",
                    "forecast_date": pd.Timestamp("2006-09-01"),
                    "horizon_hour": hour,
                    "actual_pu": actual,
                    "daylight": daylight,
                    "prediction": prediction,
                }
            )
    return pd.DataFrame(rows)


def test_metric_contract_uses_state_macro_and_zeroes_night() -> None:
    summary, per_state, site_day, horizon = evaluate_predictions(
        toy_frame(), "prediction"
    )
    assert summary["state_macro_daytime_nmae_percent"] == pytest.approx(15.0)
    assert summary["site_count"] == 2
    assert len(per_state) == 2
    assert len(site_day) == 2
    assert set(horizon["horizon_hour"]) == {1, 2}


def test_metric_contract_rejects_non_validation_rows() -> None:
    frame = toy_frame()
    frame.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="validation rows only"):
        evaluate_predictions(frame, "prediction")


def test_metric_contract_accepts_only_explicit_test_rows() -> None:
    frame = toy_frame()
    frame["split"] = "test"
    summary, *_ = evaluate_predictions(
        frame, "prediction", expected_split="test"
    )
    assert summary["split"] == "test"

    frame.loc[0, "split"] = "validation"
    with pytest.raises(ValueError, match="test rows only"):
        evaluate_predictions(
            frame, "prediction", expected_split="test"
        )
