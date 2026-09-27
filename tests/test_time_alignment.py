import pandas as pd

from paretophys_pv.time_alignment import hourly_interval_center_utc


def test_hourly_solar_reference_is_interval_center_not_left_edge() -> None:
    local = pd.DatetimeIndex(["2006-06-01 12:00:00"])
    observed = hourly_interval_center_utc(local, -6)
    expected = pd.DatetimeIndex(["2006-06-01 18:30:00"], tz="UTC")
    assert observed.equals(expected)
    assert observed[0].minute == 30


def test_each_frozen_development_offset_retains_h30_semantics() -> None:
    local = pd.DatetimeIndex(["2006-01-01 00:00:00"])
    for offset in (-8, -6, -5):
        observed = hourly_interval_center_utc(local, offset)
        assert observed[0].minute == 30
        assert observed[0].tzinfo is not None
