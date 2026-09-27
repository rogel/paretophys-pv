import numpy as np
import pytest

from paretophys_pv.data import demo_hourly, prepare_hourly
from paretophys_pv.metrics import evaluate_predictions
from paretophys_pv.search_space import Genotype
from paretophys_pv.training import predict_checkpoint, train_candidate


def test_checkpoint_roundtrip_exports_timestamps_and_preserves_metrics(tmp_path):
    data = prepare_hourly(demo_hourly())
    result = train_candidate(data, Genotype(168, "dlinear", 32, 1, 0., 12, 6, 15, 16, "scalar", .1, 0., 0., .001), tmp_path, epochs=1, device="cpu")
    frame = predict_checkpoint(data, tmp_path / "checkpoint.pt", split="validation", device="cpu")
    assert len(frame) == np.count_nonzero(data["split"] == "validation") * 24
    assert frame["local_time"].dtype.kind == "M"
    assert frame["prediction_pu"].between(0, 1.05).all()
    assert (frame.loc[~frame["daylight"], "prediction_pu"] == 0).all()
    metrics, *_ = evaluate_predictions(frame, "prediction_pu", expected_split="validation")
    assert metrics["state_macro_daytime_nmae_percent"] == pytest.approx(result["nmae_percent"], abs=1e-5)
    assert metrics["state_macro_site_day_cvar90_percent"] == pytest.approx(result["cvar90_percent"], abs=1e-5)
    data["feature_mean"][0] += 1
    with pytest.raises(ValueError, match="training scaler"):
        predict_checkpoint(data, tmp_path / "checkpoint.pt", device="cpu")
