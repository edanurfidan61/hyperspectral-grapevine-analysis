"""MLP regresyon + sınıflandırma."""

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

    def _build_mlp(in_dim, hidden, dropout, output_dim):
        layers, prev = [], in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        return nn.Sequential(*layers)


def _factory_reg(hp):
    def make(in_dim):
        return _build_mlp(in_dim, hp.get("hidden", [256, 128]),
                          float(hp.get("dropout", 0.3)), output_dim=1)
    return make


def _factory_cls(hp):
    def make(in_dim, n_classes):
        return _build_mlp(in_dim, hp.get("hidden", [256, 128]),
                          float(hp.get("dropout", 0.3)), output_dim=n_classes)
    return make


@register("regression", "mlp")
class MLPReg(BaseModel):
    is_deep_learning = True
    task = "regression"

    def _build_estimator(self):
        from sklearn.dummy import DummyRegressor
        return DummyRegressor()

    def run(self, X, y, target_name, out_dir: Path, feature_names=None, class_names=None):
        hp = self.config.get("hp", {}) or {}
        log_t = target_name in self.log_transform_targets
        return run_dl_regression(
            X=X, y=y, target_name=target_name, out_dir=out_dir, model_name="mlp",
            model_factory=_factory_reg(hp), hp=hp, random_state=self.random_state,
            conv_input=False, log_transform=log_t, sklearn_fallback=sklearn_mlp,
        )


@register("classification", "mlp")
class MLPCls(BaseModel):
    is_deep_learning = True
    task = "classification"

    def _build_estimator(self):
        from sklearn.dummy import DummyClassifier
        return DummyClassifier()

    def run(self, X, y, target_name, out_dir: Path, feature_names=None, class_names=None):
        hp = self.config.get("hp", {}) or {}
        return run_dl_classification(
            X=X, y=y, target_name=target_name, out_dir=out_dir, model_name="mlp",
            model_factory=_factory_cls(hp), hp=hp, random_state=self.random_state,
            conv_input=False, class_names=class_names, sklearn_fallback=sklearn_mlp,
        )
