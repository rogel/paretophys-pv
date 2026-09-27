"""Neural forecasting baselines with a shared day-ahead residual interface."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class DLinearBaseline(nn.Module):
    def __init__(self, lookback: int, future_features: int, horizon: int = 24) -> None:
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.seasonal = nn.Linear(lookback, horizon)
        self.trend = nn.Linear(lookback, horizon)
        self.future = nn.Linear(future_features, 1)
        self.correction = zero_module(nn.Linear(2, 1))

    def forward(
        self, history: torch.Tensor, future: torch.Tensor, base_da: torch.Tensor
    ) -> torch.Tensor:
        actual = history[..., 0]
        padded = F.pad(actual.unsqueeze(1), (12, 12), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=25, stride=1).squeeze(1)
        seasonal = actual - trend
        history_signal = self.seasonal(seasonal) + self.trend(trend)
        future_signal = self.future(future).squeeze(-1)
        correction = self.correction(
            torch.stack([history_signal, future_signal], dim=-1)
        ).squeeze(-1)
        return base_da + correction


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * 2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=dilation, padding=padding
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=dilation, padding=padding
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

    def _causal(self, value: torch.Tensor, length: int) -> torch.Tensor:
        return value[..., :length]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        length = value.shape[-1]
        residual = value
        value = self._causal(self.conv1(value), length)
        value = self.dropout(F.gelu(self.norm1(value)))
        value = self._causal(self.conv2(value), length)
        value = self.dropout(F.gelu(self.norm2(value)))
        return residual + value


class TCNBaseline(nn.Module):
    def __init__(
        self,
        history_features: int,
        future_features: int,
        channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(history_features, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[TemporalBlock(channels, dilation, dropout) for dilation in (1, 2, 4, 8)]
        )
        self.future_projection = nn.Linear(future_features, channels)
        self.head = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )
        zero_module(self.head[-1])

    def forward(
        self, history: torch.Tensor, future: torch.Tensor, base_da: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.input_projection(history.transpose(1, 2))
        encoded = self.blocks(encoded)[..., -1]
        future_encoded = self.future_projection(future)
        context = encoded.unsqueeze(1).expand(-1, future.shape[1], -1)
        correction = self.head(torch.cat([context, future_encoded], dim=-1)).squeeze(-1)
        return base_da + correction


class PatchTSTBaseline(nn.Module):
    def __init__(
        self,
        lookback: int,
        history_features: int,
        future_features: int,
        patch_length: int = 24,
        stride: int = 12,
        d_model: int = 192,
        layers: int = 4,
        heads: int = 8,
        feedforward: int = 384,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        patch_count = (lookback - patch_length) // stride + 1
        self.patch_projection = nn.Linear(patch_length * history_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, patch_count, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.future_projection = nn.Linear(future_features, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        zero_module(self.head[-1])
        nn.init.normal_(self.position, std=0.02)

    def forward(
        self, history: torch.Tensor, future: torch.Tensor, base_da: torch.Tensor
    ) -> torch.Tensor:
        patches = history.unfold(1, self.patch_length, self.stride)
        patches = patches.permute(0, 1, 3, 2).flatten(2)
        tokens = self.patch_projection(patches) + self.position
        context = self.encoder(tokens).mean(dim=1)
        future_encoded = self.future_projection(future)
        context = context.unsqueeze(1).expand(-1, future.shape[1], -1)
        correction = self.head(torch.cat([context, future_encoded], dim=-1)).squeeze(-1)
        return base_da + correction


def build_model(
    name: str, lookback: int, history_features: int, future_features: int
) -> nn.Module:
    if name == "dlinear":
        return DLinearBaseline(lookback, future_features)
    if name == "tcn":
        return TCNBaseline(history_features, future_features)
    if name == "full_patchtst":
        return PatchTSTBaseline(lookback, history_features, future_features)
    raise ValueError(f"Unknown model: {name}")
