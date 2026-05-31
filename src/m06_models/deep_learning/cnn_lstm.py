"""1D-CNN + LSTM/GRU hibrit mimarisi."""

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

    class _CNNLSTM(nn.Module):
        def __init__(self, cnn_channels, kernel_size=5, pool_every=1,
                     rnn_type="lstm", hidden_size=128, num_layers=1,
                     bidirectional=True, dropout=0.3, output_dim=1):
            super().__init__()
            layers, prev = [], 1
            for i, ch in enumerate(cnn_channels):
                layers.append(nn.Conv1d(prev, ch, kernel_size, padding=kernel_size // 2))
                layers.append(nn.BatchNorm1d(ch))
                layers.append(nn.ReLU())
                if pool_every and (i + 1) % pool_every == 0:
                    layers.append(nn.MaxPool1d(2))
                prev = ch
            self.cnn = nn.Sequential(*layers)
            cell = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
            self.rnn = cell(prev, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
            mult = 2 if bidirectional else 1
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden_size * mult, hidden_size), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_size, output_dim),
            )

        def forward(self, x):
            x = self.cnn(x).permute(0, 2, 1)
            out, _ = self.rnn(x)
            return self.head(self.dropout(out[:, -1, :]))


def _factory(hp, output_dim):
    def make(in_dim, *args):
        out = output_dim if not args else args[0]
        return _CNNLSTM(
            cnn_channels=hp.get("cnn_channels", [32, 64]),
            kernel_size=int(hp.get("kernel_size", 5)),
            pool_every=int(hp.get("pool_every", 1)),
            rnn_type=hp.get("rnn_type", "lstm"),
            hidden_size=int(hp.get("hidden_size", 128)),
            num_layers=int(hp.get("num_layers", 1)),
            bidirectional=bool(hp.get("bidirectional", True)),
            dropout=float(hp.get("dropout", 0.3)),
            output_dim=out,
        )
    return make


@register("regression", "cnn_lstm")
class CNNLSTMReg(BaseModel):
    is_deep_learning = True
    task = "regression"

    def _build_estimator(self):
        from sklearn.dummy import DummyRegressor
        return DummyRegressor()

    def run(self, X, y, target_name, out_dir: Path, feature_names=None, class_names=None):
        hp = self.config.get("hp", {}) or {}
        log_t = target_name in self.log_transform_targets
        return run_dl_regression(
            X=X, y=y, target_name=target_name, out_dir=out_dir, model_name="cnn_lstm",
            model_factory=_factory(hp, output_dim=1), hp=hp, random_state=self.random_state,
            conv_input=True, log_transform=log_t, sklearn_fallback=sklearn_mlp,
        )


@register("classification", "cnn_lstm")
class CNNLSTMCls(BaseModel):
    is_deep_learning = True
    task = "classification"

    def _build_estimator(self):
        from sklearn.dummy import DummyClassifier
        return DummyClassifier()

    def run(self, X, y, target_name, out_dir: Path, feature_names=None, class_names=None):
        hp = self.config.get("hp", {}) or {}
        return run_dl_classification(
            X=X, y=y, target_name=target_name, out_dir=out_dir, model_name="cnn_lstm",
            model_factory=_factory(hp, output_dim=None), hp=hp, random_state=self.random_state,
            conv_input=True, class_names=class_names, sklearn_fallback=sklearn_mlp,
        )
