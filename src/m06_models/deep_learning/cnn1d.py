"""1D-CNN regresyon ve sınıflandırma sarmalayıcıları."""

from __future__ import annotations

import numpy as np
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

    class _SimpleCNN(nn.Module):
        def __init__(self, in_channels, channels, kernel_size, dropout, output_dim=1):
            super().__init__()
            layers = []
            prev = in_channels
            for ch in channels:
                layers.append(nn.Conv1d(prev, ch, kernel_size, padding=kernel_size // 2))
                layers.append(nn.BatchNorm1d(ch))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool1d(2))
                prev = ch
            self.features = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(prev, output_dim))

        def forward(self, x):
            return self.head(self.pool(self.features(x)))


def _factory_reg(hp):
    def make(in_dim):
        return _SimpleCNN(
            1, hp.get("channels", [32, 64, 128]),
            int(hp.get("kernel_size", 5)), float(hp.get("dropout", 0.3)),
            output_dim=1,
        )
    return make


def _factory_cls(hp):
    def make(in_dim, n_classes):
        return _SimpleCNN(
            1, hp.get("channels", [32, 64]),
            int(hp.get("kernel_size", 5)), float(hp.get("dropout", 0.3)),
            output_dim=n_classes,
        )
    return make


@register("regression", "cnn1d")
class CNN1DReg(BaseModel):
    is_deep_learning = True
    task = "regression"

    def _build_estimator(self):
        from sklearn.dummy import DummyRegressor
        return DummyRegressor()

    def run(self, X, y, target_name, out_dir: Path, feature_names=None, class_names=None):
        hp = self.config.get("hp", {}) or {}
        log_t = target_name in self.log_transform_targets
        return run_dl_regression(
            X=X, y=y, target_name=target_name, out_dir=out_dir, model_name="cnn1d",
            model_factory=_factory_reg(hp), hp=hp, random_state=self.random_state,
            conv_input=True, log_transform=log_t, sklearn_fallback=sklearn_mlp,
        )


@register("classification", "cnn1d")
class CNN1DCls(BaseModel):
    is_deep_learning = True
    task = "classification"

    def _build_estimator(self):
        from sklearn.dummy import DummyClassifier
        return DummyClassifier()

    def run(self, X, y, target_name, out_dir: Path, feature_names=None, class_names=None):
        hp = self.config.get("hp", {}) or {}
        return run_dl_classification(
            X=X, y=y, target_name=target_name, out_dir=out_dir, model_name="cnn1d",
            model_factory=_factory_cls(hp), hp=hp, random_state=self.random_state,
            conv_input=True, class_names=class_names, sklearn_fallback=sklearn_mlp,
        )
