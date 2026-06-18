"""Flavonol için ordinal sınıflandırma (4 sınıf).

Mevcut regression/classification pipeline'a ek olarak çalışır; çıktıları
``outputs/09_ordinal_flavonol/ordinal_flavonol_report.csv`` altına yazar.

Sınıf tanımları (EMA raporundaki ≈%3.5 flavonoid düzeyi, operasyonel eşik = 3.5):
    0: flav < 1.5            (low)
    1: 1.5 <= flav < 2.5     (medium)
    2: 2.5 <= flav < 3.5     (high)
    3: flav >= 3.5           (PASS — EMA/HMPC/464682/2016 ≈%3.5 düzeyi)

Eğitilen sınıflandırıcılar (5-fold stratified CV):
    - RandomForestClassifier
    - LightGBM (yoksa RF fallback)
    - OrdinalClassifier wrapper (mord varsa LogisticAT, yoksa Frank & Hall)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core import paths
from src.core.logging_setup import get as get_logger

log = get_logger("m06_models.ordinal_flavonol")

CLASS_NAMES_4 = ["low (<1.5)", "medium [1.5,2.5)", "high [2.5,3.5)", "PASS (>=3.5)"]
CLASS_NAMES_3 = ["low (<2.5)", "high [2.5,3.5)", "PASS (>=3.5)"]


def make_ordinal_labels(y_flav: np.ndarray, n_classes: int = 4) -> np.ndarray:
    """y_flav → ordinal etiketler.

    n_classes=4: {0:<1.5, 1:[1.5,2.5), 2:[2.5,3.5), 3:>=3.5}
    n_classes=3: {0:<2.5, 1:[2.5,3.5), 2:>=3.5}  (küçük sınıfları birleştirir)
    """
    y_flav = np.asarray(y_flav, dtype=float)
    if n_classes == 4:
        labels = np.zeros(len(y_flav), dtype=int)
        labels[y_flav >= 1.5] = 1
        labels[y_flav >= 2.5] = 2
        labels[y_flav >= 3.5] = 3
        return labels
    if n_classes == 3:
        labels = np.zeros(len(y_flav), dtype=int)
        labels[y_flav >= 2.5] = 1
        labels[y_flav >= 3.5] = 2
        return labels
    raise ValueError(f"n_classes 3 veya 4 olmalı, alındı: {n_classes}")


class FrankHallOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Frank & Hall (2001) ordinal classifier wrapper.

    K sınıf için K-1 ikili sınıflandırıcı eğitir; k. model P(y > k) tahmin eder.
    Sınıf olasılıkları: P(y=0)=1-P(y>0), P(y=k)=P(y>k-1)-P(y>k), P(y=K-1)=P(y>K-2).
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        self.classes_ = np.array(sorted(np.unique(y)))
        self.n_classes_ = len(self.classes_)
        base = self.base_estimator if self.base_estimator is not None else LogisticRegression(max_iter=1000)
        self.estimators_ = []
        for k in range(self.n_classes_ - 1):
            y_bin = (y > self.classes_[k]).astype(int)
            est = clone(base)
            if len(np.unique(y_bin)) < 2:
                # tüm örnekler aynı tarafta — fit etmeyip sabit tahmin tutacağız
                self.estimators_.append(("const", int(y_bin[0])))
            else:
                est.fit(X, y_bin)
                self.estimators_.append(("model", est))
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        n = X.shape[0]
        # P(y > k) for k=0..K-2
        gt = np.zeros((n, self.n_classes_ - 1), dtype=float)
        for k, (kind, est) in enumerate(self.estimators_):
            if kind == "const":
                gt[:, k] = float(est)
            else:
                gt[:, k] = est.predict_proba(X)[:, 1]
        # P(y=k) farklarından
        proba = np.zeros((n, self.n_classes_), dtype=float)
        proba[:, 0] = 1.0 - gt[:, 0]
        for k in range(1, self.n_classes_ - 1):
            proba[:, k] = gt[:, k - 1] - gt[:, k]
        proba[:, -1] = gt[:, -1]
        # negatifleri kırp ve normalle
        proba = np.clip(proba, 0.0, None)
        row_sum = proba.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return proba / row_sum

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def _build_ordinal_classifier(random_state: int):
    """mord varsa LogisticAT; yoksa Frank & Hall LogisticRegression."""
    try:
        from mord import LogisticAT  # type: ignore

        return Pipeline([
            ("scaler", StandardScaler()),
            ("ord", LogisticAT(alpha=1.0)),
        ]), "mord.LogisticAT"
    except Exception:
        base = LogisticRegression(max_iter=1000, random_state=random_state)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ord", FrankHallOrdinalClassifier(base_estimator=base)),
        ]), "FrankHall(LogReg)"


def _build_classifiers(random_state: int) -> dict[str, tuple[object, str]]:
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    try:
        from lightgbm import LGBMClassifier  # type: ignore

        lgbm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=random_state,
                verbose=-1, verbosity=-1, force_col_wise=True,
            )),
        ])
        lgbm_label = "LightGBM"
    except Exception:
        log.warning("LightGBM yok — RF fallback kullanılıyor")
        lgbm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=500, class_weight="balanced",
                random_state=random_state, n_jobs=-1,
            )),
        ])
        lgbm_label = "RF (LGBM fallback)"

    ord_pipe, ord_label = _build_ordinal_classifier(random_state)

    return {
        "random_forest": (rf, "RandomForest"),
        "lightgbm": (lgbm, lgbm_label),
        "ordinal_logistic": (ord_pipe, ord_label),
    }


def _evaluate_fold(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    labels = list(range(n_classes))
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    mae = float(mean_absolute_error(y_true, y_pred))  # ordinal mesafe hatası
    qwk = float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic"))
    _, _, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    out = {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_f1": macro_f1,
        "mae": mae,
        "qwk": qwk,
    }
    for k in labels:
        out[f"f1_class_{k}"] = float(f1_per[k])
    return out


def _load_data() -> tuple[np.ndarray, np.ndarray]:
    ds_dir = paths.OUTPUTS_DIR / "01_dataset"
    X = np.load(ds_dir / "X.npy")
    y_flav = np.load(ds_dir / "y_flav.npy")
    return X, y_flav


def _run_one_scheme(
    Xv: np.ndarray,
    yv: np.ndarray,
    n_classes: int,
    cv_splits: int,
    random_state: int,
) -> list[dict]:
    """Tek bir sınıf şeması (3 veya 4 sınıf) için tüm modelleri CV ile değerlendir."""
    y_ord = make_ordinal_labels(yv, n_classes=n_classes)
    log.info("Şema=%d sınıf | dağılım=%s",
             n_classes, {int(c): int((y_ord == c).sum()) for c in range(n_classes)})

    counts = np.bincount(y_ord, minlength=n_classes)
    cv = cv_splits
    if counts.min() < cv:
        new_cv = max(2, int(counts[counts > 0].min()))
        log.warning("  cv düşürülüyor: %d → %d (en küçük sınıf=%d)",
                    cv, new_cv, int(counts.min()))
        cv = new_cv

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    classifiers = _build_classifiers(random_state)

    rows: list[dict] = []
    for model_key, (estimator, label) in classifiers.items():
        log.info("  Model: %s (%s)", model_key, label)
        fold_metrics: list[dict] = []
        for fold_idx, (tr, te) in enumerate(skf.split(Xv, y_ord), start=1):
            est = clone(estimator)
            est.fit(Xv[tr], y_ord[tr])
            y_hat = est.predict(Xv[te])
            m = _evaluate_fold(y_ord[te], y_hat, n_classes=n_classes)
            m.update({"scheme": f"{n_classes}cls", "model": model_key,
                      "label": label, "fold": fold_idx})
            fold_metrics.append(m)
            rows.append(m)
            log.info("    fold %d: acc=%.3f bacc=%.3f macroF1=%.3f mae=%.3f qwk=%.3f",
                     fold_idx, m["accuracy"], m["balanced_accuracy"],
                     m["macro_f1"], m["mae"], m["qwk"])

        mean_row = {"scheme": f"{n_classes}cls", "model": model_key,
                    "label": label, "fold": "mean"}
        keys = [k for k in fold_metrics[0].keys()
                if k not in ("scheme", "model", "label", "fold")]
        for k in keys:
            mean_row[k] = float(np.mean([fm[k] for fm in fold_metrics]))
        rows.append(mean_row)
        log.info("    MEAN: acc=%.3f bacc=%.3f macroF1=%.3f mae=%.3f qwk=%.3f",
                 mean_row["accuracy"], mean_row["balanced_accuracy"],
                 mean_row["macro_f1"], mean_row["mae"], mean_row["qwk"])
    return rows


def run(cfg=None) -> Path:
    """Pipeline aşaması: ordinal flavonol sınıflandırması (3 + 4 sınıf şeması)."""
    cv_splits = int(cfg.get("models.cv", 5)) if cfg is not None else 5
    random_state = int(cfg.get("models.random_state", 42)) if cfg is not None else 42

    X, y_flav = _load_data()
    valid = np.isfinite(y_flav)
    Xv, yv = X[valid], y_flav[valid]
    log.info("Ordinal flavonol: n=%d", len(yv))

    rows: list[dict] = []
    rows += _run_one_scheme(Xv, yv, n_classes=4, cv_splits=cv_splits, random_state=random_state)
    rows += _run_one_scheme(Xv, yv, n_classes=3, cv_splits=cv_splits, random_state=random_state)

    # her iki şemada da var olan ortak sütunlar + sınıf-başı F1 (max 4)
    base_cols = ["scheme", "model", "label", "fold",
                 "accuracy", "balanced_accuracy", "macro_f1", "mae", "qwk"]
    f1_cols = [f"f1_class_{k}" for k in range(4)]
    df = pd.DataFrame(rows)
    for c in f1_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[base_cols + f1_cols]

    # Tek kaynak: numaralı aşama dizini (eski outputs/reports/ yazımı kaldırıldı).
    stage_dir = paths.stage_dir("09_ordinal_flavonol")
    out_csv = stage_dir / "ordinal_flavonol_report.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    log.info("Rapor yazıldı: %s", out_csv)

    paths.write_source_marker(
        stage_dir,
        producer="src/m06_models/ordinal_flavonol.py",
        config_source=cfg.source if cfg is not None else None,
    )
    return out_csv


if __name__ == "__main__":
    run()
