"""Shared timestamp semantics for hourly PV aggregation and solar geometry."""

from __future__ import annotations

import pandas as pd


HOURLY_INTERVAL_CENTER_MINUTES = 30


def hourly_interval_center_utc(
    local_hour_start: pd.DatetimeIndex,
    utc_offset_hours: int,
) -> pd.DatetimeIndex:
    """Map local H:00 interval labels to the H:30 physical reference in UTC."""

    return pd.DatetimeIndex(
        local_hour_start
        + pd.Timedelta(minutes=HOURLY_INTERVAL_CENTER_MINUTES)
        - pd.Timedelta(hours=int(utc_offset_hours)),
        tz="UTC",
    )
