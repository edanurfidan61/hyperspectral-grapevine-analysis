"""Denoising Autoencoder (DAE) → linear probe (regresyon/sınıflandırma).

Pipeline:
    Spektral X → gürültü → Encoder → Latent → Decoder → X̂ ≈ X
                            ↓
                       Probe → çıktı
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

from src.m06_models.registry import register
from src.m06_models.base import BaseModel
from src.m06_models.utils import (
    regression_metrics,
    classify_metrics,
    save_metrics_json,
    plot_learning_curve,
    plot_regression,
    plot_residuals,
    plot_confusion,
)
from src.core.logging_setup import get as get_logger

log = get_logger("m06_models.deep_learning.autoencoder")

try:
    import torch
    import torch.nn as nn
    from src.m06_models.deep_learning.trainer import Trainer
    from src.m06_models.deep_learning.datasets import (
        make_loader, prepare_dl_data, prepare_dl_kfolds,
    )
    TORCH = True
except Exception:
    TORCH = False


if TORCH:
    class _Encoder(nn.Module):
        def __init__(self, in_dim, hidden, latent):
            super().__init__()
            layers, prev = [], in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, latent))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


    class _Decoder(nn.Module):
        def __init__(self, latent, hidden, out_dim):
            super().__init__()
            layers, prev = [], latent
            for h in reversed(hidden):
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, z):
            return self.net(z)


    class _DAE(nn.Module):
        def __init__(self, in_dim, hidden, latent, noise_std=0.15):
            super().__init__()
            self.encoder = _Encoder(in_dim, hidden, latent)
            self.decoder = _Decoder(latent, hidden, in_dim)
            self.noise_std = float(noise_std)

        def forward(self, x):
            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            return self.decoder(self.encoder(x))


    class _Probe(nn.Module):
        def __init__(self, encoder, latent, output_dim, dropout=0.2):
            super().__init__()
            self.encoder = encoder
            mid = max(latent // 2, 8)
            self.head = nn.Sequential(
                nn.Linear(latent, mid), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(mid, output_dim),
            )

        def forward(self, x):
            return self.head(self.encoder(x))


def _trainer_kwargs(hp, *, epochs_key="epochs", patience_default=15):
    return dict(
        device=hp.get("device", None),
        epochs=int(hp.get(epochs_key, 100)),
        batch_size=int(hp.get("batch_size", 32)),
        lr=float(hp.get("lr", 1e-3)),
        patience=int(hp.get("patience", patience_default)),
        weight_decay=float(hp.get("weight_decay", 0.0)),
        amp=bool(hp.get("amp", False)),
    )


def _aug_kwargs(hp):
    return dict(
        augment_noise=float(hp.get("augment_noise", 0.0)),
        augment_shift=int(hp.get("augment_shift", 0)),
    )


def _sklearn_fallback(X_train, y_train, X_val, regression: bool, **hp):
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    hidden = tuple(hp.get("hidden", (128, 64)))
    cls = MLPRegressor if regression else MLPClassifier
    clf = cls(hidden_layer_sizes=hidden, max_iter=int(hp.get("epochs", 200)))
    clf.fit(X_train, y_train)
    return clf.predict(X_val)


def _train_one_fold(
    X_tr, X_va, y_tr, y_va, *, hp, regression: bool, n_classes: int, out_dir: Path
):
    in_dim = X_tr.shape[1]
    hidden = list(hp.get("hidden", [128, 64]))
    latent = int(hp.get("latent_dim", min(32, in_dim // 2)))
    noise_std = float(hp.get("noise_std", 0.15))

    # Pretrain (denoising)
    ae = _DAE(in_dim, hidden, latent, noise_std=noise_std)
    tr1 = Trainer(ae, **_trainer_kwargs(hp, epochs_key="pretrain_epochs"))
    rl_t = make_loader(X_tr, X_tr, batch_size=tr1.batch_size, shuffle=True)
    rl_v = make_loader(X_va, X_va, batch_size=tr1.batch_size, shuffle=False)
    tr1.fit(rl_t, rl_v, criterion=nn.MSELoss(), checkpoint_path=out_dir / "ae_ckpt.pt")

    # Probe
    out_dim = 1 if regression else n_classes
    probe = _Probe(ae.encoder, latent, output_dim=out_dim,
                   dropout=float(hp.get("dropout", 0.2)))
    tr2 = Trainer(probe, **_trainer_kwargs(hp, epochs_key="finetune_epochs", patience_default=10))
    aug = _aug_kwargs(hp)
    tl = make_loader(X_tr, y_tr, batch_size=tr2.batch_size, shuffle=True, augment_kwargs=aug)
    vl = make_loader(X_va, y_va, batch_size=tr2.batch_size, shuffle=False)

    if regression:
        criterion = nn.MSELoss()
    else:
        from src.m06_models.deep_learning.runner import _class_weights
        w = torch.tensor(_class_weights(y_tr, n_classes), device=tr2.device)
        criterion = nn.CrossEntropyLoss(weight=w)

    history = tr2.fit(tl, vl, criterion=criterion, checkpoint_path=out_dir / "probe_ckpt.pt")
    preds = tr2.predict(vl)
    if not regression:
        preds = np.argmax(preds, axis=1) if preds.ndim > 1 else (preds >= 0.5).astype(int)
    return preds, history


@register("regression", "autoencoder")
class AEReg(BaseModel):
    is_deep_learning = True
    task = "regression"

    def _build_estimator(self):
        from sklearn.dummy import DummyRegressor
        return DummyRegressor()

    def run(self, X, y, target_name, out_dir: Path, feature_names=None, class_names=None):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        hp = self.config.get("hp", {}) or {}
        val_size = float(hp.get("val_size", 0.2))
        n_splits = int(hp.get("n_splits", 1))
        log_t = target_name in self.log_transform_targets

        y_t = np.log1p(np.maximum(np.asarray(y, dtype=np.float64), -1.0 + 1e-9)) if log_t else np.asarray(y)

        if not TORCH:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            valid = np.isfinite(y_t)
            X_tr, X_va, y_tr, y_va = train_test_split(
                np.asarray(X)[valid], y_t[valid], test_size=val_size, random_state=self.random_state
            )
            sc = StandardScaler().fit(X_tr)
            preds = _sklearn_fallback(sc.transform(X_tr), y_tr, sc.transform(X_va), regression=True, **hp)
            if log_t:
                preds = np.expm1(preds); y_va = np.expm1(y_va)
            metrics = regression_metrics(y_va, preds)
            save_metrics_json(metrics, out_dir / "metrics.json")
            return {"metrics": metrics, "y_true": y_va, "y_pred": preds}

        if n_splits > 1:
            all_pred, all_true = [], []
            history = {"train_loss": [], "val_loss": []}
            for X_tr, X_va, y_tr, y_va, _ in prepare_dl_kfolds(
                X, y_t, regression=True, n_splits=n_splits, random_state=self.random_state
            ):
                preds, history = _train_one_fold(
                    X_tr, X_va, y_tr, y_va, hp=hp, regression=True, n_classes=1, out_dir=out_dir
                )
                if log_t:
                    preds = np.expm1(preds); y_va = np.expm1(y_va)
                all_pred.append(preds); all_true.append(y_va)
            preds = np.concatenate(all_pred); y_val_all = np.concatenate(all_true)
        else:
            X_tr, X_va, y_tr, y_va, _ = prepare_dl_data(
                X, y_t, regression=True, val_size=val_size, random_state=self.random_state
            )
            preds, history = _train_one_fold(
                X_tr, X_va, y_tr, y_va, hp=hp, regression=True, n_classes=1, out_dir=out_dir
            )
            if log_t:
                preds = np.expm1(preds); y_va = np.expm1(y_va)
            y_val_all = y_va

        metrics = regression_metrics(y_val_all, preds)
        save_metrics_json(metrics, out_dir / "metrics.json")
        try:
            plot_learning_curve(history, out_dir / "learning_curve.png")
        except Exception:
            pass
        try:
            plot_regression(y_val_all, preds, out_dir / "scatter.png", title=f"autoencoder | {target_name}")
            plot_residuals(y_val_all, preds, out_dir / "residuals.png", title=f"autoencoder | {target_name}")
        except Exception:
            pass
        return {"metrics": metrics, "y_true": y_val_all, "y_pred": preds}


@register("classification", "autoencoder")
class AECls(BaseModel):
    is_deep_learning = True
    task = "classification"

    def _build_estimator(self):
        from sklearn.dummy import DummyClassifier
        return DummyClassifier()

    def run(self, X, y, target_name, out_dir: Path, feature_names=None, class_names=None):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        hp = self.config.get("hp", {}) or {}
        val_size = float(hp.get("val_size", 0.2))
        n_splits = int(hp.get("n_splits", 1))
        n_classes = int(max(int(np.asarray(y).max()) + 1, 2))

        if not TORCH:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            y_arr = np.asarray(y).astype(int)
            X_tr, X_va, y_tr, y_va = train_test_split(
                np.asarray(X), y_arr, test_size=val_size, random_state=self.random_state, stratify=y_arr
            )
            sc = StandardScaler().fit(X_tr)
            preds = _sklearn_fallback(sc.transform(X_tr), y_tr, sc.transform(X_va), regression=False, **hp)
            metrics = classify_metrics(y_va, preds, class_names=class_names)
            save_metrics_json(metrics, out_dir / "metrics.json")
            return {"metrics": metrics, "y_true": y_va, "y_pred": preds}

        if n_splits > 1:
            all_pred, all_true = [], []
            history = {"train_loss": [], "val_loss": []}
            for X_tr, X_va, y_tr, y_va, _ in prepare_dl_kfolds(
                X, y, regression=False, n_splits=n_splits, random_state=self.random_state
            ):
                preds, history = _train_one_fold(
                    X_tr, X_va, y_tr, y_va, hp=hp, regression=False, n_classes=n_classes, out_dir=out_dir
                )
                all_pred.append(preds); all_true.append(y_va)
            preds = np.concatenate(all_pred); y_val_all = np.concatenate(all_true)
        else:
            X_tr, X_va, y_tr, y_va, _ = prepare_dl_data(
                X, y, regression=False, val_size=val_size, random_state=self.random_state
            )
            preds, history = _train_one_fold(
                X_tr, X_va, y_tr, y_va, hp=hp, regression=False, n_classes=n_classes, out_dir=out_dir
            )
            y_val_all = y_va

        metrics = classify_metrics(y_val_all, preds, class_names=class_names)
        save_metrics_json(metrics, out_dir / "metrics.json")
        try:
            plot_learning_curve(history, out_dir / "learning_curve.png")
        except Exception:
            pass
        try:
            plot_confusion(metrics["confusion_matrix"], out_dir / "confusion.png",
                           class_names=class_names, title=f"autoencoder | {target_name}")
        except Exception:
            pass
        return {"metrics": metrics, "y_true": y_val_all, "y_pred": preds}
