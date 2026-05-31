"""1D-ResNet sarmalayıcıları."""

from __future__ import annotations

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
    import torch.nn as nn

    class _ResBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3):
            super().__init__()
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
            self.bn1 = nn.BatchNorm1d(out_ch)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
            self.bn2 = nn.BatchNorm1d(out_ch)
            self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

        def forward(self, x):
            out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
            out = self.conv2(out); out = self.bn2(out)
            if self.down is not None:
                x = self.down(x)
            return self.relu(out + x)


    class _ResNet1D(nn.Module):
        def __init__(self, in_ch, channels, kernel_size=3, dropout=0.3, output_dim=1):
            super().__init__()
            blocks, prev = [], in_ch
            for ch in channels:
                blocks.append(_ResBlock(prev, ch, kernel_size=kernel_size)); prev = ch
            self.blocks = nn.Sequential(*blocks)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(prev, output_dim))

        def forward(self, x):
            return self.head(self.pool(self.blocks(x)))


def _factory_reg(hp):
    def make(in_dim):
        return _ResNet1D(
            1, hp.get("channels", [32, 64, 128]),
            kernel_size=int(hp.get("kernel_size", 3)),
            dropout=float(hp.get("dropout", 0.3)), output_dim=1,
        )
    return make


def _factory_cls(hp):
    def make(in_dim, n_classes):
        return _ResNet1D(
            1, hp.get("channels", [32, 64]),
            kernel_size=int(hp.get("kernel_size", 3)),
            dropout=float(hp.get("dropout", 0.3)), output_dim=n_classes,
        )
    return make


@register("regression", "resnet1d")
class ResNet1DReg(BaseModel):
    is_deep_learning = True
    task = "regression"

    def _build_estimator(self):
        from sklearn.dummy import DummyRegressor
        return DummyRegressor()

    def run(self, X, y, target_name, out_dir: Path, feature_names=None, class_names=None):
        hp = self.config.get("hp", {}) or {}
        log_t = target_name in self.log_transform_targets
        return run_dl_regression(
            X=X, y=y, target_name=target_name, out_dir=out_dir, model_name="resnet1d",
            model_factory=_factory_reg(hp), hp=hp, random_state=self.random_state,
            conv_input=True, log_transform=log_t, sklearn_fallback=sklearn_mlp,
        )


@register("classification", "resnet1d")
class ResNet1DCls(BaseModel):
    is_deep_learning = True
    task = "classification"

    def _build_estimator(self):
        from sklearn.dummy import DummyClassifier
        return DummyClassifier()

    def run(self, X, y, target_name, out_dir: Path, feature_names=None, class_names=None):
        hp = self.config.get("hp", {}) or {}
        return run_dl_classification(
            X=X, y=y, target_name=target_name, out_dir=out_dir, model_name="resnet1d",
            model_factory=_factory_cls(hp), hp=hp, random_state=self.random_state,
            conv_input=True, class_names=class_names, sklearn_fallback=sklearn_mlp,
        )
