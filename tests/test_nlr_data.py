from paretophys_pv.nlr_data import Site, deterministic_farthest_point_sites, parse_member


def test_parse_interval_schema() -> None:
    assert parse_member(
        "tx", "DA_32.05_-94.15_2006_UPV_95MW_60_Min.csv"
    ).interval_minutes == 60


def test_parse_fractional_capacity_member() -> None:
    member = parse_member(
        "or", "Actual_42.15_-121.75_2006_UPV_0.5MW_5_Min.csv"
    )
    assert member.series == "Actual"
    assert member.interval_minutes == 5
    assert member.site.capacity_mw == 0.5
    assert member.site.longitude == -121.75


def test_farthest_point_is_deterministic_and_capped() -> None:
    sites = [
        Site("x", 0.0, 0.0, "UPV", 1.0),
        Site("x", 1.0, 0.0, "UPV", 1.0),
        Site("x", 0.0, 1.0, "UPV", 1.0),
        Site("x", 1.0, 1.0, "UPV", 1.0),
    ]
    first = deterministic_farthest_point_sites(sites, 3)
    second = deterministic_farthest_point_sites(reversed(sites), 3)
    assert first == second
    assert len(first) == 3
    assert first[0].key == min(site.key for site in sites)


def test_farthest_point_uses_all_sites_when_below_target() -> None:
    sites = [
        Site("or", 42.0, -122.0, "UPV", 1.0),
        Site("or", 43.0, -123.0, "UPV", 1.0),
    ]
    assert len(deterministic_farthest_point_sites(sites, 20)) == 2
