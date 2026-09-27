"""Mixed-variable search space and bounded residual PV models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from paretophys_pv.features import FEATURE_COLUMNS


LOOKBACK_VALUES = (48, 72, 168)
BACKBONE_VALUES = ("dlinear", "tcn", "compact_patchtst")
HIDDEN_VALUES = (32, 64, 96)
LAYER_VALUES = (1, 2, 3)
DROPOUT_VALUES = (0.0, 0.1, 0.2)
PATCH_VALUES = (12, 24)
STRIDE_VALUES = (6, 12)
RESIDUAL_WIDTH_VALUES = (16, 32, 64)
GATE_VALUES = ("scalar", "horizon_wise")
CORRECTION_CAP_VALUES = (0.05, 0.10, 0.20)
LAMBDA_R_VALUES = (0.0, 0.1, 0.3)
LAMBDA_S_VALUES = (0.0, 0.01)
LEARNING_RATE_VALUES = (1e-3, 3e-4, 1e-4)
COMPLEXITY_ESTIMATOR_VERSION = "pytorch_flop_counter_train_graph_v2"

STATE_GROUP_INDICES = {
    "S1": tuple(FEATURE_COLUMNS.index(name) for name in (
        "solar_proxy", "clear_sky_ghi_norm", "daylight_float"
    )),
    "S2": tuple(FEATURE_COLUMNS.index(name) for name in (
        "da_pu", "da_increment_pu", "da_peak_pu", "da_peak_hour_sin",
        "da_peak_hour_cos", "da_energy_puh"
    )),
    "S3": tuple(FEATURE_COLUMNS.index(name) for name in (
        "previous_day_same_hour_pu", "trailing_7_day_same_hour_mean_pu",
        "origin_trailing_actual_mean_24h_pu", "origin_trailing_actual_mean_72h_pu",
        "origin_trailing_abs_ramp_mean_24h_pu", "origin_trailing_abs_ramp_mean_72h_pu"
    )),
    "S4": tuple(FEATURE_COLUMNS.index(name) for name in (
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "latitude", "longitude"
    )),
}
MODEL_FEATURE_INDICES = tuple(sorted({i for values in STATE_GROUP_INDICES.values() for i in values}))


@dataclass(frozen=True)
class Genotype:
    lookback: int
    backbone: str
    hidden_width: int
    layers: int
    dropout: float
    patch_length: int
    stride: int
    state_mask: int
    residual_width: int
    gate: str
    correction_cap: float
    lambda_r: float
    lambda_s: float
    learning_rate: float

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    def key(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RepairResult:
    genotype: Genotype
    actions: tuple[str, ...]
    parameter_count: int
    fp32_size_mb: float
    constraint_violation: float


def sample_genotype(rng: np.random.Generator) -> Genotype:
    return Genotype(
        lookback=int(rng.choice(LOOKBACK_VALUES)),
        backbone=str(rng.choice(BACKBONE_VALUES)),
        hidden_width=int(rng.choice(HIDDEN_VALUES)),
        layers=int(rng.choice(LAYER_VALUES)),
        dropout=float(rng.choice(DROPOUT_VALUES)),
        patch_length=int(rng.choice(PATCH_VALUES)),
        stride=int(rng.choice(STRIDE_VALUES)),
        state_mask=int(rng.integers(0, 16)),
        residual_width=int(rng.choice(RESIDUAL_WIDTH_VALUES)),
        gate=str(rng.choice(GATE_VALUES)),
        correction_cap=float(rng.choice(CORRECTION_CAP_VALUES)),
        lambda_r=float(rng.choice(LAMBDA_R_VALUES)),
        lambda_s=float(rng.choice(LAMBDA_S_VALUES)),
        learning_rate=float(rng.choice(LEARNING_RATE_VALUES)),
    )


def state_indices(mask: int) -> tuple[int, ...]:
    values: list[int] = []
    for bit, name in enumerate(("S1", "S2", "S3", "S4")):
        if mask & (1 << bit):
            values.extend(STATE_GROUP_INDICES[name])
    return tuple(sorted(set(values)))


def _zero(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class DLinearBackbone(nn.Module):
    def __init__(self, genotype: Genotype, future_features: int) -> None:
        super().__init__()
        self.lookback = genotype.lookback
        self.seasonal = nn.Linear(genotype.lookback, 24)
        self.trend = nn.Linear(genotype.lookback, 24)
        modules: list[nn.Module] = [nn.Linear(future_features, genotype.hidden_width), nn.GELU()]
        for _ in range(genotype.layers - 1):
            modules.extend([
                nn.Linear(genotype.hidden_width, genotype.hidden_width),
                nn.GELU(),
                nn.Dropout(genotype.dropout),
            ])
        self.future = nn.Sequential(*modules)
        self.head = _zero(nn.Linear(genotype.hidden_width + 1, 1))

    def forward(self, history: torch.Tensor, future: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        actual = history[..., 0]
        padded = F.pad(actual.unsqueeze(1), (12, 12), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=25, stride=1).squeeze(1)
        seasonal = actual - trend
        signal = self.seasonal(seasonal) + self.trend(trend)
        encoded = self.future(future)
        correction = self.head(torch.cat([signal.unsqueeze(-1), encoded], dim=-1)).squeeze(-1)
        return base + correction


class CausalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = 2 * dilation
        self.conv1 = nn.Conv1d(width, width, 3, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, width)
        self.norm2 = nn.GroupNorm(1, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        length = value.shape[-1]
        residual = value
        value = self.conv1(value)[..., :length]
        value = self.dropout(F.gelu(self.norm1(value)))
        value = self.conv2(value)[..., :length]
        value = self.dropout(F.gelu(self.norm2(value)))
        return residual + value


class TCNBackbone(nn.Module):
    def __init__(self, genotype: Genotype, history_features: int, future_features: int) -> None:
        super().__init__()
        width = genotype.hidden_width
        self.input_projection = nn.Conv1d(history_features, width, 1)
        self.blocks = nn.Sequential(*[
            CausalBlock(width, 2**index, genotype.dropout)
            for index in range(genotype.layers)
        ])
        self.future = nn.Linear(future_features, width)
        self.head = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(genotype.dropout),
            nn.Linear(width, 1),
        )
        _zero(self.head[-1])

    def forward(self, history: torch.Tensor, future: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        context = self.blocks(self.input_projection(history.transpose(1, 2)))[..., -1]
        context = context.unsqueeze(1).expand(-1, future.shape[1], -1)
        correction = self.head(torch.cat([context, self.future(future)], dim=-1)).squeeze(-1)
        return base + correction


class CompactPatchTSTBackbone(nn.Module):
    def __init__(self, genotype: Genotype, history_features: int, future_features: int) -> None:
        super().__init__()
        self.patch_length = genotype.patch_length
        self.stride = genotype.stride
        patch_count = (genotype.lookback - genotype.patch_length) // genotype.stride + 1
        width = genotype.hidden_width
        self.patch_projection = nn.Linear(genotype.patch_length * history_features, width)
        self.position = nn.Parameter(torch.zeros(1, patch_count, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 2,
            dropout=genotype.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=genotype.layers)
        self.future = nn.Linear(future_features, width)
        self.head = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(genotype.dropout),
            nn.Linear(width, 1),
        )
        _zero(self.head[-1])
        nn.init.normal_(self.position, std=0.02)

    def forward(self, history: torch.Tensor, future: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        patches = history.unfold(1, self.patch_length, self.stride)
        patches = patches.permute(0, 1, 3, 2).flatten(2)
        context = self.encoder(self.patch_projection(patches) + self.position).mean(dim=1)
        context = context.unsqueeze(1).expand(-1, future.shape[1], -1)
        correction = self.head(torch.cat([context, self.future(future)], dim=-1)).squeeze(-1)
        return base + correction


class PhysicsResidualAdapter(nn.Module):
    def __init__(self, genotype: Genotype) -> None:
        super().__init__()
        indices = state_indices(genotype.state_mask)
        if not indices:
            raise ValueError("At least one state group must be enabled")
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long), persistent=True)
        width = genotype.residual_width
        self.encoder = nn.Sequential(
            nn.Linear(len(indices), width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.residual = _zero(nn.Linear(width, 1))
        self.gate = nn.Linear(width, 1)
        self.gate_type = genotype.gate
        self.cap = genotype.correction_cap

    def forward(self, future_all: torch.Tensor, daylight: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(torch.index_select(future_all, -1, self.indices))
        residual = torch.tanh(self.residual(encoded).squeeze(-1))
        if self.gate_type == "scalar":
            weights = daylight.float().unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            gate = torch.sigmoid(self.gate(pooled)).expand(-1, future_all.shape[1])
        else:
            gate = torch.sigmoid(self.gate(encoded).squeeze(-1))
        return self.cap * daylight.float() * gate * residual


class ParetoPhysPV(nn.Module):
    def __init__(self, genotype: Genotype, history_features: int = 2) -> None:
        super().__init__()
        self.genotype = genotype
        self.register_buffer(
            "model_feature_indices", torch.tensor(MODEL_FEATURE_INDICES, dtype=torch.long), persistent=True
        )
        future_features = len(MODEL_FEATURE_INDICES)
        if genotype.backbone == "dlinear":
            self.backbone = DLinearBackbone(genotype, future_features)
        elif genotype.backbone == "tcn":
            self.backbone = TCNBackbone(genotype, history_features, future_features)
        elif genotype.backbone == "compact_patchtst":
            self.backbone = CompactPatchTSTBackbone(genotype, history_features, future_features)
        else:
            raise ValueError(f"Unknown backbone: {genotype.backbone}")
        self.adapter = PhysicsResidualAdapter(genotype)

    def forward(
        self,
        history: torch.Tensor,
        future_all: torch.Tensor,
        base_da: torch.Tensor,
        daylight: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history = history[:, -self.genotype.lookback :]
        future = torch.index_select(future_all, -1, self.model_feature_indices)
        backbone_prediction = self.backbone(history, future, base_da)
        physical_correction = self.adapter(future_all, daylight)
        total_correction = backbone_prediction - base_da + physical_correction
        prediction = torch.clamp(base_da + total_correction, 0.0, 1.05)
        prediction = torch.where(daylight, prediction, torch.zeros_like(prediction))
        if return_aux:
            return prediction, {
                "backbone_prediction": backbone_prediction,
                "physical_correction": physical_correction,
                "total_correction": total_correction,
            }
        return prediction


def build_model(genotype: Genotype) -> ParetoPhysPV:
    return ParetoPhysPV(genotype)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_macs(model: ParetoPhysPV) -> int:
    # TransformerEncoder uses a fused inference fast path under eval+no_grad
    # that hides attention operators from FlopCounterMode. Count the same
    # non-fused graph used for proxy training so all three backbones share a
    # complete, deterministic operator accounting path.
    previous_training = model.training
    model = model.cpu().train()
    genotype = model.genotype
    history = torch.zeros(1, genotype.lookback, 2)
    future = torch.zeros(1, 24, len(FEATURE_COLUMNS))
    base = torch.zeros(1, 24)
    daylight = torch.ones(1, 24, dtype=torch.bool)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(history, future, base, daylight)
    macs = max(1, int(math.ceil(counter.get_total_flops() / 2.0)))
    model.train(previous_training)
    return macs


def repair_genotype(raw: Genotype) -> RepairResult:
    values = raw.canonical_dict()
    actions: list[str] = []
    if values["state_mask"] & 0b0011 == 0:
        values["state_mask"] |= 0b0001
        actions.append("enable_S1_future_state")
    if values["backbone"] != "compact_patchtst":
        if values["patch_length"] != 12 or values["stride"] != 6:
            actions.append("canonicalize_inactive_patch_variables")
        values["patch_length"] = 12
        values["stride"] = 6
    else:
        if values["patch_length"] > values["lookback"]:
            values["patch_length"] = max(value for value in PATCH_VALUES if value <= values["lookback"])
            actions.append("reduce_patch_length_to_lookback")
        if values["stride"] > values["patch_length"]:
            values["stride"] = max(value for value in STRIDE_VALUES if value <= values["patch_length"])
            actions.append("reduce_stride_to_patch_length")
    if values["correction_cap"] > 0.20:
        values["correction_cap"] = 0.20
        actions.append("cap_correction_at_0_20")

    genotype = Genotype(**values)
    parameters = count_parameters(build_model(genotype))
    while parameters > 1_000_000 and genotype.hidden_width > min(HIDDEN_VALUES):
        smaller = max(value for value in HIDDEN_VALUES if value < genotype.hidden_width)
        values["hidden_width"] = smaller
        actions.append(f"reduce_hidden_width_to_{smaller}")
        genotype = Genotype(**values)
        parameters = count_parameters(build_model(genotype))
    while parameters > 1_000_000 and genotype.layers > min(LAYER_VALUES):
        values["layers"] -= 1
        actions.append(f"reduce_layers_to_{values['layers']}")
        genotype = Genotype(**values)
        parameters = count_parameters(build_model(genotype))

    fp32_mb = parameters * 4 / 1_000_000
    violation = max(0.0, (parameters - 1_000_000) / 1_000_000)
    violation += max(0.0, (fp32_mb - 10.0) / 10.0)
    violation += max(0.0, genotype.correction_cap - 0.20)
    return RepairResult(genotype, tuple(actions), parameters, fp32_mb, violation)


def proxy_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    daylight: torch.Tensor,
    base_da: torch.Tensor,
    total_correction: torch.Tensor,
    lambda_r: float,
    lambda_s: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = daylight.float()
    day_mae = ((prediction - target).abs() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    pair_mask = (daylight[:, 1:] | daylight[:, :-1]).float()
    predicted_ramp = prediction[:, 1:] - prediction[:, :-1]
    target_ramp = target[:, 1:] - target[:, :-1]
    ramp = ((predicted_ramp - target_ramp).abs() * pair_mask).sum(dim=1)
    ramp = ramp / pair_mask.sum(dim=1).clamp_min(1.0)
    correction_second = torch.diff(total_correction, n=2, dim=1).abs()
    base_second = torch.diff(base_da, n=2, dim=1).abs()
    triple_mask = (daylight[:, 2:] | daylight[:, 1:-1] | daylight[:, :-2]).float()
    smooth = (F.relu(correction_second - base_second) * triple_mask).sum(dim=1)
    smooth = smooth / triple_mask.sum(dim=1).clamp_min(1.0)
    components = {
        "day_mae": day_mae.mean(),
        "ramp": ramp.mean(),
        "smooth": smooth.mean(),
    }
    total = components["day_mae"] + lambda_r * components["ramp"] + lambda_s * components["smooth"]
    return total, components
