import numpy as np
import torch

from paretophys_pv.features import FEATURE_COLUMNS
from paretophys_pv.search_space import (
    Genotype,
    build_model,
    count_macs,
    count_parameters,
    proxy_loss,
    repair_genotype,
    sample_genotype,
    state_indices,
)


def genotype(**overrides) -> Genotype:
    values = {
        "lookback": 72,
        "backbone": "tcn",
        "hidden_width": 32,
        "layers": 2,
        "dropout": 0.1,
        "patch_length": 24,
        "stride": 12,
        "state_mask": 0b1111,
        "residual_width": 16,
        "gate": "horizon_wise",
        "correction_cap": 0.1,
        "lambda_r": 0.1,
        "lambda_s": 0.01,
        "learning_rate": 1e-3,
    }
    values.update(overrides)
    return Genotype(**values)


def test_sampling_is_deterministic() -> None:
    first = sample_genotype(np.random.default_rng(2202))
    second = sample_genotype(np.random.default_rng(2202))
    assert first == second
    assert first.key() == second.key()


def test_repair_is_deterministic_and_enables_future_state() -> None:
    raw = genotype(state_mask=0b1100, correction_cap=0.3)
    first = repair_genotype(raw)
    second = repair_genotype(raw)
    assert first == second
    assert first.genotype.state_mask & 0b0011
    assert first.genotype.correction_cap == 0.2
    assert first.genotype.patch_length == 12
    assert first.genotype.stride == 6
    assert first.constraint_violation == 0.0
    assert first.parameter_count <= 1_000_000


def test_state_groups_never_include_capacity_or_state_identity() -> None:
    indices = state_indices(0b1111)
    excluded = {
        FEATURE_COLUMNS.index("log1p_capacity_mw"),
        FEATURE_COLUMNS.index("state_ny"),
        FEATURE_COLUMNS.index("state_or"),
        FEATURE_COLUMNS.index("state_tx"),
    }
    assert not excluded.intersection(indices)


def test_all_backbones_respect_bounds_night_and_complexity() -> None:
    history = torch.randn(2, 168, 2)
    future = torch.randn(2, 24, len(FEATURE_COLUMNS))
    base = torch.rand(2, 24)
    daylight = torch.zeros(2, 24, dtype=torch.bool)
    daylight[:, 7:19] = True
    for name in ("dlinear", "tcn", "compact_patchtst"):
        repaired = repair_genotype(genotype(backbone=name, hidden_width=64, layers=2))
        model = build_model(repaired.genotype).eval()
        with torch.no_grad():
            prediction = model(history, future, base, daylight)
        assert prediction.shape == (2, 24)
        assert torch.all(prediction >= 0.0)
        assert torch.all(prediction <= 1.05)
        assert torch.all(prediction[~daylight] == 0.0)
        assert count_parameters(model) == repaired.parameter_count
        first_macs = count_macs(model)
        second_macs = count_macs(model)
        assert first_macs > 0
        assert first_macs == second_macs


def test_proxy_loss_is_finite_and_differentiable() -> None:
    repaired = repair_genotype(genotype()).genotype
    model = build_model(repaired)
    history = torch.randn(2, 168, 2)
    future = torch.randn(2, 24, len(FEATURE_COLUMNS))
    base = torch.rand(2, 24)
    target = torch.rand(2, 24)
    daylight = torch.zeros(2, 24, dtype=torch.bool)
    daylight[:, 6:20] = True
    prediction, auxiliary = model(history, future, base, daylight, return_aux=True)
    loss, components = proxy_loss(
        prediction,
        target,
        daylight,
        base,
        auxiliary["total_correction"],
        repaired.lambda_r,
        repaired.lambda_s,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert set(components) == {"day_mae", "ramp", "smooth"}
    assert any(parameter.grad is not None for parameter in model.parameters())
