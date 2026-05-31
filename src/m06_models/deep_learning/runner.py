"""DL modelleri için ortak eğitim/değerlendirme runner'ı.

Tek-split veya K-fold CV (config: ``n_splits``), augmentation, sınıf-ağırlıklı
loss, target log-transform, sklearn fallback — hepsi tek noktada.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from src.core.logging_setup import get as get_logger
from src.m06_models.utils import (
    classify_metrics,
    plot_confusion,
    plot_learning_curve,
    plot_regression,
    plot_residuals,
    regression_metrics,
    save_metrics_json,
)

log = get_logger("m06_models.deep_learning.runner")

try:
    import torch
    import torch.nn as nn
    from src.m06_models.deep_learning.trainer import Trainer
    from src.m06_models.deep_learning.datasets import (
        make_loader,
        prepare_dl_data,
        prepare_dl_kfolds,
    )
    TORCH = True
except Exception:
    TORCH = False


def trainer_kwargs(hp: dict) -> dict:
    return dict(
        device=hp.get("device", None),
        epochs=int(hp.get("epochs", 100)),
        batch_size=int(hp.get("batch_size", 32)),
        lr=float(hp.get("lr", 1e-3)),
        patience=int(hp.get("patience", 15)),
        weight_decay=float(hp.get("weight_decay", 0.0)),
        amp=bool(hp.get("amp", False)),
    )


def _aug_kwargs(hp: dict) -> dict:
    return dict(
        augment_noise=float(hp.get("augment_noise", 0.0)),
        augment_shift=int(hp.get("augment_shift", 0)),
    )


def _class_weights(y_train: np.ndarray, n_classes: int) -> np.ndarray:
    classes, counts = np.unique(y_train, return_counts=True)
    w = np.ones(n_classes, dtype=np.float32)
    for c, cnt in zip(classes, counts):
        w[int(c)] = len(y_train) / (len(classes) * max(int(cnt), 1))
    return w


def _reshape(X: np.ndarray, conv: bool) -> np.ndarray:
    """conv=True ise (B, 1, L); değilse (B, L) tutar."""
    if conv:
        return X.reshape(-1, 1, X.shape[1])
    return X


def run_dl_regression(
    *,
    X: np.ndarray,
    y: np.ndarray,
    target_name: str,
    out_dir: Path,
    model_name: str,
    model_factory: Callable[[int], "nn.Module"],
    hp: dict,
    random_state: int,
    conv_input: bool,
    log_transform: bool = False,
    sklearn_fallback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """DL regresyon: tek-split veya k-fold + scaling + augmentation."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    val_size = float(hp.get("val_size", 0.2))
    n_splits = int(hp.get("n_splits", 1))
    aug = _aug_kwargs(hp)

    if log_transform:
        y_t = np.log1p(np.maximum(np.asarray(y, dtype=np.float64), -1.0 + 1e-9))
    else:
        y_t = np.asarray(y, dtype=np.float64)

    if not TORCH:
        if sklearn_fallback is None:
            raise RuntimeError("PyTorch yok ve sklearn fallback verilmedi")
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        valid = np.isfinite(y_t)
        X_tr, X_va, y_tr, y_va = train_test_split(
            np.asarray(X)[valid], y_t[valid], test_size=val_size, random_state=random_state
        )
        sc = StandardScaler().fit(X_tr)
        preds = sklearn_fallback(sc.transform(X_tr), y_tr, sc.transform(X_va), regression=True, **hp)
        if log_transform:
            preds = np.expm1(preds); y_va = np.expm1(y_va)
        metrics = regression_metrics(y_va, preds)
        save_metrics_json(metrics, out_dir / "metrics.json")
        return {"metrics": metrics, "y_true": y_va, "y_pred": preds}

    if n_splits > 1:
        all_pred, all_true = [], []
        history = {"train_loss": [], "val_loss": []}
        for X_tr, X_va, y_tr, y_va, _ in prepare_dl_kfolds(
            X, y_t, regression=True, n_splits=n_splits, random_state=random_state
        ):
            in_dim = X_tr.shape[1]
            X_tr_in = _reshape(X_tr, conv_input)
            X_va_in = _reshape(X_va, conv_input)
            model = model_factory(in_dim)
            tr = Trainer(model, **trainer_kwargs(hp))
            tl = make_loader(X_tr_in, y_tr, batch_size=tr.batch_size, shuffle=True, augment_kwargs=aug)
            vl = make_loader(X_va_in, y_va, batch_size=tr.batch_size, shuffle=False)
            history = tr.fit(tl, vl, criterion=nn.MSELoss(), checkpoint_path=out_dir / "ckpt.pt")
            preds = tr.predict(vl)
            if log_transform:
                preds = np.expm1(preds); y_va = np.expm1(y_va)
            all_pred.append(preds); all_true.append(y_va)
        preds = np.concatenate(all_pred); y_val_all = np.concatenate(all_true)
    else:
        X_tr, X_va, y_tr, y_va, _ = prepare_dl_data(
            X, y_t, regression=True, val_size=val_size, random_state=random_state
        )
        in_dim = X_tr.shape[1]
        X_tr_in = _reshape(X_tr, conv_input)
        X_va_in = _reshape(X_va, conv_input)
        model = model_factory(in_dim)
        tr = Trainer(model, **trainer_kwargs(hp))
        tl = make_loader(X_tr_in, y_tr, batch_size=tr.batch_size, shuffle=True, augment_kwargs=aug)
        vl = make_loader(X_va_in, y_va, batch_size=tr.batch_size, shuffle=False)
        history = tr.fit(tl, vl, criterion=nn.MSELoss(), checkpoint_path=out_dir / "ckpt.pt")
        preds = tr.predict(vl)
        if log_transform:
            preds = np.expm1(preds); y_va = np.expm1(y_va)
        y_val_all = y_va

    metrics = regression_metrics(y_val_all, preds)
    save_metrics_json(metrics, out_dir / "metrics.json")
    try:
        plot_learning_curve(history, out_dir / "learning_curve.png")
    except Exception:
        pass
    try:
        plot_regression(y_val_all, preds, out_dir / "scatter.png", title=f"{model_name} | {target_name}")
        plot_residuals(y_val_all, preds, out_dir / "residuals.png", title=f"{model_name} | {target_name}")
    except Exception:
        pass
    return {"metrics": metrics, "y_true": y_val_all, "y_pred": preds}


def run_dl_classification(
    *,
    X: np.ndarray,
    y: np.ndarray,
    target_name: str,
    out_dir: Path,
    model_name: str,
    model_factory: Callable[[int, int], "nn.Module"],
    hp: dict,
    random_state: int,
    conv_input: bool,
    class_names: Optional[list] = None,
    sklearn_fallback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """DL sınıflandırma: tek-split veya k-fold, sınıf-ağırlıklı CE loss."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    val_size = float(hp.get("val_size", 0.2))
    n_splits = int(hp.get("n_splits", 1))
    aug = _aug_kwargs(hp)

    if not TORCH:
        if sklearn_fallback is None:
            raise RuntimeError("PyTorch yok ve sklearn fallback verilmedi")
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        y_arr = np.asarray(y).astype(int)
        X_tr, X_va, y_tr, y_va = train_test_split(
            np.asarray(X), y_arr, test_size=val_size, random_state=random_state, stratify=y_arr
        )
        sc = StandardScaler().fit(X_tr)
        preds = sklearn_fallback(sc.transform(X_tr), y_tr, sc.transform(X_va), regression=False, **hp)
        metrics = classify_metrics(y_va, preds, class_names=class_names)
        save_metrics_json(metrics, out_dir / "metrics.json")
        return {"metrics": metrics, "y_true": y_va, "y_pred": preds}

    def _train_one(X_tr, X_va, y_tr, y_va):
        in_dim = X_tr.shape[1]
        n_classes = int(max(int(np.asarray(y).max()) + 1, 2))
        X_tr_in = _reshape(X_tr, conv_input)
        X_va_in = _reshape(X_va, conv_input)
        model = model_factory(in_dim, n_classes)
        tr = Trainer(model, **trainer_kwargs(hp))
        tl = make_loader(X_tr_in, y_tr, batch_size=tr.batch_size, shuffle=True, augment_kwargs=aug)
        vl = make_loader(X_va_in, y_va, batch_size=tr.batch_size, shuffle=False)
        weights = torch.tensor(_class_weights(y_tr, n_classes), device=tr.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        history = tr.fit(tl, vl, criterion=criterion, checkpoint_path=out_dir / "ckpt.pt")
        preds_raw = tr.predict(vl)
        preds = np.argmax(preds_raw, axis=1) if preds_raw.ndim > 1 else (preds_raw >= 0.5).astype(int)
        return preds, history

    if n_splits > 1:
        all_pred, all_true = [], []
        history = {"train_loss": [], "val_loss": []}
        for X_tr, X_va, y_tr, y_va, _ in prepare_dl_kfolds(
            X, y, regression=False, n_splits=n_splits, random_state=random_state
        ):
            preds, history = _train_one(X_tr, X_va, y_tr, y_va)
            all_pred.append(preds); all_true.append(y_va)
        preds = np.concatenate(all_pred); y_val_all = np.concatenate(all_true)
    else:
        X_tr, X_va, y_tr, y_va, _ = prepare_dl_data(
            X, y, regression=False, val_size=val_size, random_state=random_state
        )
        preds, history = _train_one(X_tr, X_va, y_tr, y_va)
        y_val_all = y_va

    metrics = classify_metrics(y_val_all, preds, class_names=class_names)
    save_metrics_json(metrics, out_dir / "metrics.json")
    try:
        plot_learning_curve(history, out_dir / "learning_curve.png")
    except Exception:
        pass
    try:
        plot_confusion(metrics["confusion_matrix"], out_dir / "confusion.png",
                       class_names=class_names, title=f"{model_name} | {target_name}")
    except Exception:
        pass
    return {"metrics": metrics, "y_true": y_val_all, "y_pred": preds}


def sklearn_mlp(X_train, y_train, X_val, regression: bool, **hp):
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    hidden = tuple(hp.get("hidden", (128, 64)))
    cls = MLPRegressor if regression else MLPClassifier
    clf = cls(hidden_layer_sizes=hidden, max_iter=int(hp.get("epochs", 200)))
    clf.fit(X_train, y_train)
    return clf.predict(X_val)
