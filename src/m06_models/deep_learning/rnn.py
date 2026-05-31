"""LSTM/GRU sarmalayıcıları (regresyon + sınıflandırma)."""

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

    class _RNNNet(nn.Module):
        def __init__(self, input_size=1, hidden_size=128, num_layers=1, rnn_type="lstm",
                     bidirectional=False, dropout=0.3, output_dim=1):
            super().__init__()
            cell = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
            self.rnn = cell(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
            mult = 2 if bidirectional else 1
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size * mult, output_dim)

        def forward(self, x):
            if x.dim() == 3 and x.shape[1] == 1:
                x = x.permute(0, 2, 1)
            out, _ = self.rnn(x)
            return self.head(self.dropout(out[:, -1, :]))


def _factory_reg(hp):
    def make(in_dim):
        return _RNNNet(
            input_size=1,
            hidden_size=int(hp.get("hidden_size", 128)),
            num_layers=int(hp.get("num_layers", 1)),
            rnn_type=hp.get("type", "lstm"),
            bidirectional=bool(hp.get("bidirectional", False)),
            dropout=float(hp.get("dropout", 0.3)),
            output_dim=1,
        )
    return make


def _factory_cls(hp):
    def make(in_dim, n_classes):
        return _RNNNet(
            input_size=1,
            hidden_size=int(hp.get("hidden_size", 128)),
            num_layers=int(hp.get("num_layers", 1)),
            rnn_type=hp.get("type", "lstm"),
            bidirectional=bool(hp.get("bidirectional", False)),
            dropout=float(hp.get("dropout", 0.3)),
            output_dim=n_classes,
        )
    return make


@register("regression", "rnn")
class RNNReg(BaseModel):
    is_deep_learning = True
    task = "regression"

    def _build_estimator(self):
        from sklearn.dummy import DummyRegressor
        return DummyRegressor()

    def run(self, X, y, target_name, out_dir: Path, feature_names=None, class_names=None):
        hp = self.config.get("hp", {}) or {}
        log_t = target_name in self.log_transform_targets
        return run_dl_regression(
            X=X, y=y, target_name=target_name, out_dir=out_dir, model_name="rnn",
            model_factory=_factory_reg(hp), hp=hp, random_state=self.random_state,
            conv_input=True, log_transform=log_t, sklearn_fallback=sklearn_mlp,
        )


@register("classification", "rnn")
class RNNCls(BaseModel):
    is_deep_learning = True
    task = "classification"

    def _build_estimator(self):
        from sklearn.dummy import DummyClassifier
        return DummyClassifier()

    def run(self, X, y, target_name, out_dir: Path, feature_names=None, class_names=None):
        hp = self.config.get("hp", {}) or {}
        return run_dl_classification(
            X=X, y=y, target_name=target_name, out_dir=out_dir, model_name="rnn",
            model_factory=_factory_cls(hp), hp=hp, random_state=self.random_state,
            conv_input=True, class_names=class_names, sklearn_fallback=sklearn_mlp,
        )
