"""Pareto coverage screening and preference-based configuration selection."""

import numpy as np

OBJECTIVES = ("nmae_percent", "cvar90_percent", "log10_macs")


def unique_records(records):
    unique = {}
    for row in records:
        key = row["key"]
        if key in unique and any(not np.isclose(row[name], unique[key][name], rtol=0, atol=1e-10) for name in OBJECTIVES):
            raise ValueError("Repeated configurations have different objectives; aggregate repeated training runs first")
        unique[key] = row
    return [unique[key] for key in sorted(unique)]


def pareto_records(records):
    records = unique_records(records)
    values = np.asarray([[r[name] for name in OBJECTIVES] for r in records], dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Selection requires nonempty finite objective values")
    return [row for i, row in enumerate(records) if not np.any(np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1))]


def assign_roles(records, accuracy_tolerance=.02):
    """Normalize over the supplied evaluation population, including dominated records."""
    if accuracy_tolerance < 0:
        raise ValueError("Accuracy tolerance must be nonnegative")
    records = unique_records(records)
    if not records:
        raise ValueError("Selection requires at least one configuration")
    values = np.asarray([[r[name] for name in OBJECTIVES] for r in records], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Objectives must be finite")
    spans = np.ptp(values, axis=0)
    normalized = (values - values.min(axis=0)) / np.where(spans > 0, spans, 1.)
    distances = {row["key"]: float(np.linalg.norm(normalized[i])) for i, row in enumerate(records)}
    tie = lambda row: (row["nmae_percent"], row["cvar90_percent"], row["log10_macs"], row["key"])
    accuracy = min(records, key=tie)
    balanced = min(records, key=lambda row: (distances[row["key"]], *tie(row)))
    eligible = [r for r in records if r["nmae_percent"] <= accuracy["nmae_percent"] * (1 + accuracy_tolerance) + 1e-12]
    efficiency = min(eligible, key=lambda row: (row["log10_macs"], *tie(row)))
    return {"accuracy": accuracy, "balanced": balanced, "efficiency": efficiency}


def screen_candidates(records, count=8):
    if count < 4:
        raise ValueError("Coverage screening needs at least four slots for its anchors")
    pool = pareto_records(records)
    roles = assign_roles(pool)
    tail = min(pool, key=lambda r: (r["cvar90_percent"], r["nmae_percent"], r["log10_macs"], r["key"]))
    anchors = [roles["accuracy"], tail, roles["balanced"], roles["efficiency"]]
    selected = list({r["key"]: r for r in anchors}.values())
    values = np.asarray([[r[name] for name in OBJECTIVES] for r in pool])
    spans = np.ptp(values, axis=0)
    normalized = (values - values.min(axis=0)) / np.where(spans > 0, spans, 1.)
    by_key = {r["key"]: normalized[i] for i, r in enumerate(pool)}
    while len(selected) < min(count, len(pool)):
        remaining = [r for r in pool if r["key"] not in {s["key"] for s in selected}]
        distance = lambda r: min(float(np.linalg.norm(by_key[r["key"]] - by_key[s["key"]])) for s in selected)
        selected.append(min(remaining, key=lambda r: (-distance(r), r["key"])))
    return selected
