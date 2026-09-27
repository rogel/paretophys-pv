"""Parsing and deterministic site selection for the NLR PV archives."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np


NLR_FILENAME = re.compile(
    r"^(?P<series>Actual|DA|HA4)_"
    r"(?P<latitude>-?\d+(?:\.\d+)?)_"
    r"(?P<longitude>-?\d+(?:\.\d+)?)_"
    r"(?P<year>\d{4})_"
    r"(?P<pv_type>UPV|DPV)_"
    r"(?P<capacity_mw>\d+(?:\.\d+)?)MW_"
    r"(?P<interval_minutes>5|60)_Min\.csv$"
)


@dataclass(frozen=True, order=True)
class Site:
    state: str
    latitude: float
    longitude: float
    pv_type: str
    capacity_mw: float

    @property
    def key(self) -> str:
        return (
            f"{self.state}:{self.latitude:.6f}:{self.longitude:.6f}:"
            f"{self.pv_type}:{self.capacity_mw:g}MW"
        )


@dataclass(frozen=True)
class Member:
    filename: str
    series: str
    year: int
    interval_minutes: int
    site: Site


def parse_member(state: str, filename: str) -> Member:
    """Parse an archive member and reject names outside the archive naming schema."""

    match = NLR_FILENAME.fullmatch(filename)
    if match is None:
        raise ValueError(f"Unexpected NLR archive member: {filename}")
    data = match.groupdict()
    return Member(
        filename=filename,
        series=data["series"],
        year=int(data["year"]),
        interval_minutes=int(data["interval_minutes"]),
        site=Site(
            state=state,
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
            pv_type=data["pv_type"],
            capacity_mw=float(data["capacity_mw"]),
        ),
    )


def deterministic_farthest_point_sites(
    sites: Iterable[Site], target: int
) -> list[Site]:
    """Select up to ``target`` sites without looking at forecasting errors.

    Coordinates are standardized within each state. Selection starts from the
    lexicographically smallest site key and repeatedly maximizes distance to the
    nearest selected site; lexical order breaks exact ties.
    """

    ordered = sorted(set(sites), key=lambda site: site.key)
    if target <= 0 or not ordered:
        return []
    if len(ordered) <= target:
        return ordered

    coordinates = np.asarray(
        [[site.latitude, site.longitude] for site in ordered], dtype=np.float64
    )
    scale = coordinates.std(axis=0)
    scale[scale == 0.0] = 1.0
    standardized = (coordinates - coordinates.mean(axis=0)) / scale

    selected_indices = [0]
    remaining = set(range(1, len(ordered)))
    nearest_sq = np.sum((standardized - standardized[0]) ** 2, axis=1)

    while remaining and len(selected_indices) < target:
        next_index = min(
            remaining,
            key=lambda index: (-nearest_sq[index], ordered[index].key),
        )
        selected_indices.append(next_index)
        remaining.remove(next_index)
        distance_sq = np.sum(
            (standardized - standardized[next_index]) ** 2, axis=1
        )
        nearest_sq = np.minimum(nearest_sq, distance_sq)

    return [ordered[index] for index in selected_indices]
