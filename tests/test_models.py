import torch

from paretophys_pv.models import build_model


def test_all_neural_baselines_start_from_raw_da() -> None:
    history = torch.randn(2, 168, 2)
    future = torch.randn(2, 24, 25)
    base = torch.rand(2, 24)
    for name in ("dlinear", "tcn", "full_patchtst"):
        model = build_model(name, 168, 2, 25).eval()
        with torch.no_grad():
            prediction = model(history, future, base)
        assert prediction.shape == (2, 24)
        assert torch.allclose(prediction, base)


def test_all_neural_baselines_can_move_away_from_raw_da() -> None:
    torch.manual_seed(7)
    history = torch.randn(2, 168, 2)
    future = torch.randn(2, 24, 25)
    base = torch.rand(2, 24)
    target = torch.rand(2, 24)
    for name in ("dlinear", "tcn", "full_patchtst"):
        model = build_model(name, 168, 2, 25).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad()
            prediction = model(history, future, base)
            loss = torch.abs(prediction - target).mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            prediction = model.eval()(history, future, base)
        assert not torch.allclose(prediction, base)
