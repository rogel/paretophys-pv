from paretophys_pv.selection import assign_roles, pareto_records, screen_candidates


def row(key, mean, tail, cost):
    return {"key": key, "nmae_percent": mean, "cvar90_percent": tail, "log10_macs": cost}


def test_dominated_configuration_is_excluded():
    rows = [row("a", 1, 3, 4), row("b", 2, 2, 3), row("c", 3, 4, 5)]
    assert {r["key"] for r in pareto_records(rows)} == {"a", "b"}


def test_zero_objective_span_and_single_candidate_are_supported():
    rows = [row("a", 1, 3, 4)]
    assert {r["key"] for r in assign_roles(rows).values()} == {"a"}
    assert screen_candidates(rows) == rows


def test_efficiency_role_obeys_accuracy_tolerance():
    rows = [row("accurate", 10, 20, 7), row("efficient", 10.1, 21, 5), row("too_inaccurate", 11, 22, 4)]
    roles = assign_roles(rows)
    assert roles["accuracy"]["key"] == "accurate"
    assert roles["efficiency"]["key"] == "efficient"


def test_screening_keeps_all_four_distinct_anchors():
    rows = [row("a", 10, 25, 6), row("b", 11, 15, 6), row("c", 10.5, 19, 5.5), row("d", 10.1, 26, 4)]
    assert {r["key"] for r in screen_candidates(rows, 4)} == {"a", "b", "c", "d"}
