"""Training, validation, and checkpoint inference for bounded residual models."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .data import ARRAY_NAMES, META_NAMES
from .features import FEATURE_COLUMNS
from .search_space import Genotype, build_model, count_macs, count_parameters, proxy_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    return torch.device(requested)


def objective_metrics(prediction, target, daylight, states) -> tuple[float, float]:
    prediction = np.clip(prediction, 0., 1.05)
    errors = np.abs(prediction - target)
    means, tails = [], []
    for state in sorted(np.unique(states)):
        selected = states == state
        mask = daylight[selected]
        means.append(float(errors[selected][mask].mean() * 100))
        daily = (errors[selected] * mask).sum(axis=1) / mask.sum(axis=1)
        count = max(1, int(np.ceil(len(daily) * .1)))
        tails.append(float(np.sort(daily)[-count:].mean() * 100))
    return float(np.mean(means)), float(np.mean(tails))


def make_loader(data, indices, batch_size, sampler=None):
    tensors = [torch.from_numpy(data[name][indices]) for name in ARRAY_NAMES]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0)


@torch.no_grad()
def predict_array(model, loader, device) -> np.ndarray:
    model.eval()
    result = []
    for history, future, base, _, daylight in loader:
        result.append(model(history.to(device), future.to(device), base.to(device), daylight.to(device)).cpu().numpy())
    return np.concatenate(result)


def train_candidate(data, genotype: Genotype, output: str | Path, *, epochs=40,
                    patience=6, batch_size=256, seed=2026, device="auto") -> dict:
    if min(epochs, patience, batch_size) < 1:
        raise ValueError("Epochs, patience, and batch size must be positive")
    train_indices = np.flatnonzero(data["split"] == "train")
    validation_indices = np.flatnonzero(data["split"] == "validation")
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Training requires separate train and validation samples")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoint.pt").exists():
        raise FileExistsError(f"Use a new output directory; {output / 'checkpoint.pt'} already exists")
    device = choose_device(device)
    set_seed(seed)
    model = build_model(genotype)
    parameters, macs = count_parameters(model), count_macs(model)
    set_seed(seed)
    model = model.to(device)
    index = pd.DataFrame({name: data[name][train_indices] for name in META_NAMES})
    sites_per_state = index.groupby("state").site_key.nunique()
    samples_per_site = index.groupby("site_key").size()
    weights = np.asarray([1 / (len(sites_per_state) * sites_per_state[row.state] * samples_per_site[row.site_key]) for row in index.itertuples()])
    sampler = WeightedRandomSampler(torch.as_tensor(weights / weights.mean(), dtype=torch.double), len(index), replacement=True, generator=torch.Generator().manual_seed(seed))
    train_loader = make_loader(data, train_indices, batch_size, sampler)
    validation_loader = make_loader(data, validation_indices, max(batch_size, 512))
    optimizer = torch.optim.AdamW(model.parameters(), lr=genotype.learning_rate, weight_decay=1e-4)
    best_score, stale, best_epoch, best_state = float("inf"), 0, 0, None
    history_rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for history, future, base, target, daylight in train_loader:
            history, future, base, target, daylight = [x.to(device) for x in (history, future, base, target, daylight)]
            optimizer.zero_grad(set_to_none=True)
            prediction, auxiliary = model(history, future, base, daylight, return_aux=True)
            loss, _ = proxy_loss(prediction, target, daylight, base, auxiliary["total_correction"], genotype.lambda_r, genotype.lambda_s)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        pred = predict_array(model, validation_loader, device)
        score, tail = objective_metrics(pred, data["target"][validation_indices], data["daylight"][validation_indices], data["state"][validation_indices])
        history_rows.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "validation_nmae_percent": score, "validation_cvar90_percent": tail})
        if score < best_score - 1e-4:
            best_score, stale, best_epoch = score, 0, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was produced")
    model.load_state_dict(best_state)
    pred = predict_array(model, validation_loader, device)
    nmae, tail = objective_metrics(pred, data["target"][validation_indices], data["daylight"][validation_indices], data["state"][validation_indices])
    checkpoint = {"genotype": genotype.canonical_dict(), "state_dict": best_state,
                  "feature_mean": data["feature_mean"].tolist(), "feature_std": data["feature_std"].tolist(),
                  "feature_names": FEATURE_COLUMNS, "training_seed": seed}
    torch.save(checkpoint, output / "checkpoint.pt")
    result = {"key": genotype.key(), "genotype": genotype.canonical_dict(), "nmae_percent": nmae,
              "cvar90_percent": tail, "log10_macs": float(np.log10(macs)), "macs": macs,
              "parameters": parameters, "best_epoch": best_epoch, "epochs_run": len(history_rows),
              "seed": seed, "device": str(device), "checkpoint": str(output / "checkpoint.pt")}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(history_rows).to_csv(output / "history.csv", index=False)
    return result


def predict_checkpoint(data, checkpoint: str | Path, *, split="test", device="auto", batch_size=256):
    indices = np.flatnonzero(data["split"] == split)
    if not len(indices):
        raise ValueError(f"No samples in split {split}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["feature_names"] != FEATURE_COLUMNS:
        raise ValueError("Checkpoint feature order is incompatible")
    for name in ("feature_mean", "feature_std"):
        if not np.array_equal(np.asarray(saved[name], dtype=np.float32), data[name]):
            raise ValueError("Bundle must use the checkpoint's training scaler")
    model = build_model(Genotype(**saved["genotype"]))
    model.load_state_dict(saved["state_dict"])
    device = choose_device(device)
    pred = predict_array(model.to(device), make_loader(data, indices, batch_size), device)
    rows = []
    for i, sequence in enumerate(indices):
        origin = pd.Timestamp(str(data["forecast_origin"][sequence]))
        rows.append(pd.DataFrame({"state": data["state"][sequence], "site_key": data["site_key"][sequence],
            "local_time": pd.date_range(origin, periods=24, freq="h"), "split": split,
            "forecast_date": origin.normalize(), "horizon_hour": np.arange(24),
            "actual_pu": data["target"][sequence], "daylight": data["daylight"][sequence], "prediction_pu": pred[i]}))
    return pd.concat(rows, ignore_index=True)
