import numpy as np
import pytest

from paretophys_pv.data import demo_hourly, load_bundle, prepare_hourly, save_bundle


def test_scaler_uses_training_period_only():
    frame = demo_hourly()
    original = prepare_hourly(frame)
    frame.loc[frame["split"].isin(["validation", "test"]), "da_pu"] *= 2
    altered = prepare_hourly(frame)
    np.testing.assert_array_equal(original["feature_mean"], altered["feature_mean"])
    np.testing.assert_array_equal(original["feature_std"], altered["feature_std"])
    train = original["split"] == "train"
    np.testing.assert_array_equal(original["future"][train], altered["future"][train])


def test_sequence_bundle_roundtrip_and_shapes(tmp_path):
    data = prepare_hourly(demo_hourly())
    output = tmp_path / "bundle.npz"
    save_bundle(output, data)
    observed = load_bundle(output)
    assert observed["history"].shape == (46, 168, 2)
    assert observed["future"].shape == (46, 24, 25)
    for name in data:
        np.testing.assert_array_equal(observed[name], data[name])


def test_missing_hour_is_rejected():
    frame = demo_hourly().drop(index=10)
    with pytest.raises(ValueError, match="discontinuous"):
        prepare_hourly(frame)


def test_overlap_between_chronological_splits_is_rejected():
    data = prepare_hourly(demo_hourly())
    i = np.flatnonzero(data["split"] == "test")[0]
    data["split"][i] = "train"
    from paretophys_pv.data import validate_bundle
    with pytest.raises(ValueError, match="chronological"):
        validate_bundle(data)
