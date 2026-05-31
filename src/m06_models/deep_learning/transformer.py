"""Küçük Transformer encoder sarmalayıcısı."""

from __future__ import annotations

import math
from pathlib import Path

from src.m06_models.registry import register
from src.m06_models.base import BaseModel
from src.m06_models.deep_learning.runner import (
    run_dl_regression,
    run_dl_classification,
    sklearn_mlp,
    TORCH,
)

if TORCH:
    import torch
    import torch.nn as nn

    class _PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=4096):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]


    class _Transformer1D(nn.Module):
        def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                     dropout=0.1, output_dim=1):
            super().__init__()
            self.input_proj = nn.Linear(1, d_model)
            self.posenc = _PositionalEncoding(d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, output_dim)

        def forward(self, x):
            if x.dim() == 3 and x.shape[1] == 1:
                x = x.permute(0, 2, 1)
            x = self.input_proj(x)
            x = self.posenc(x)
            x = self.encoder(x)
            x = self.norm(x.mean(dim=1))
            return self.head(x)


def _factory_reg(hp):
    def make(in_dim):
        return _Transformer1D(
            d_model=int(hp.get("d_model", 64)),
            nhead=int(hp.get("nhead", 4)),
            num_layers=int(hp.get("num_layers", 2)),
            dim_feedforward=int(hp.get("dim_feedforward", 128)),
            dropout=float(hp.get("dropout", 0.1)),
            output_dim=1,
        )
    return make


def _factory_cls(hp):
    def make(in_dim, n_classes):
        return _Transformer1D(
            d_model=int(hp.get("d_model", 64)),
            nhead=int(hp.get("nhead", 4)),
            num_layers=int(hp.get("num_layers", 2)),
            dim_feedforward=int(hp.get("dim_feedforward", 128)),
            dropout=float(hp.get("dropout", 0.1)),
            output_dim=n_classes,
        )
    return make


@register("regression", "transformer")
class TransformerReg(BaseModel):
    is_deep_learning = True
    task = "regression"

    def _build_estimator(self):
        from sklearn.dummy import DummyRegressor
        return DummyRegressor()

    def run(self, X, y, target_name, out_dir: Path, feature_names=None, class_names=None):
        hp = self.config.get("hp", {}) or {}
        log_t = target_name in self.log_transform_targets
        return run_dl_regression(
            X=X, y=y, target_name=target_name, out_dir=out_dir, model_name="transformer",
            model_factory=_factory_reg(hp), hp=hp, random_state=self.random_state,
            conv_input=True, log_transform=log_t, sklearn_fallback=sklearn_mlp,
        )


@register("classification", "transformer")
class TransformerCls(BaseModel):
    is_deep_learning = True
    task = "classification"

    def _build_estimator(self):
        from sklearn.dummy import DummyClassifier
        return DummyClassifier()

    def run(self, X, y, target_name, out_dir: Path, feature_names=None, class_names=None):
        hp = self.config.get("hp", {}) or {}
        return run_dl_classification(
            X=X, y=y, target_name=target_name, out_dir=out_dir, model_name="transformer",
            model_factory=_factory_cls(hp), hp=hp, random_state=self.random_state,
            conv_input=True, class_names=class_names, sklearn_fallback=sklearn_mlp,
        )
