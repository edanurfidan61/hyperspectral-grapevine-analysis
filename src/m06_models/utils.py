"""Modeller arası ortak metrik ve görselleştirme yardımcıları.

Bu modül eski projedeki ``module_6/utils_model.py``'ın temizlenmiş halidir;
``BaseModel`` ve concrete model sınıfları buradaki fonksiyonları kullanır.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

from src.core.logging_setup import get as _get_logger

_log = _get_logger("m06_models.utils")


# ---------------------------------------------------------------------------
# Dengesizlik (resampling) yardımcısı — GÖREV 1
# ---------------------------------------------------------------------------
def apply_smote_tomek(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    method: str = "smote_tomek",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sınıf dengesizliğini gidermek için over-/under-sampling uygula.

    Parameters
    ----------
    X_train, y_train
        SADECE eğitim fold'unun verisi. Validation/test verisine asla
        uygulanmamalıdır (leakage yasak).
    method
        ``"none"`` → resampling kapalı; X, y olduğu gibi döner.
        ``"smote"`` → SMOTE (sentetik over-sampling).
        ``"smote_tomek"`` → SMOTE + Tomek Links (over + under birlikte).
    random_state
        SMOTE/Tomek için RNG seed'i.

    Notes
    -----
    - ``k_neighbors`` dinamik seçilir:
      ``min_count = en küçük sınıf - 1; k = max(1, min(5, min_count))``.
      Bu, mildiyö / downy mildew (2 örnek) gibi çok küçük sınıflarda SMOTE'un kırılmasını önler.
    - Bir sınıfta 1 örnek varsa SMOTE skip edilir (en az 2 örnek gerekir);
      orijinal X,y döner, uyarı log'lanır.
    """
    if method == "none":
        return X_train, y_train

    from collections import Counter
    counts = Counter(np.asarray(y_train).tolist())
    if len(counts) < 2:
        _log.warning("Resampling skip: tek sınıf var (n=%d)", len(y_train))
        return X_train, y_train

    min_count = min(counts.values())
    if min_count < 2:
        _log.warning(
            "Resampling skip: en küçük sınıfta yalnız %d örnek (SMOTE ≥2 ister) "
            "— sınıf dağılımı: %s", min_count, dict(counts),
        )
        return X_train, y_train

    # k_neighbors = SMOTE komşu sayısı; en küçük sınıftan büyük olamaz
    k = max(1, min(5, min_count - 1))

    from imblearn.over_sampling import SMOTE
    smote = SMOTE(k_neighbors=k, random_state=random_state)

    if method == "smote":
        X_res, y_res = smote.fit_resample(X_train, y_train)
    elif method == "smote_tomek":
        from imblearn.combine import SMOTETomek
        from imblearn.under_sampling import TomekLinks
        sampler = SMOTETomek(
            smote=smote, tomek=TomekLinks(), random_state=random_state,
        )
        X_res, y_res = sampler.fit_resample(X_train, y_train)
    else:
        raise ValueError(
            f"Bilinmeyen resampling method: {method!r}. "
            "Geçerli: 'none' | 'smote' | 'smote_tomek'"
        )

    _log.debug(
        "Resampling[%s] k=%d: %d→%d örnek (sınıf dağılımı %s → %s)",
        method, k, len(y_train), len(y_res),
        dict(counts), dict(Counter(np.asarray(y_res).tolist())),
    )
    return X_res, y_res


class SafeResampler:
    """imblearn Pipeline ile uyumlu, dinamik-k SMOTE sarmalayıcı.

    GÖREV 2: RandomizedSearchCV içinde fold-içi SMOTE uygularken k_neighbors'ı
    dinamik seçen ``apply_smote_tomek``'i kullanırız; aksi takdirde küçük
    sınıfların bulunduğu fold'larda SMOTE varsayılan k=5 ile çöker.
    """

    def __init__(self, method: str = "smote_tomek", random_state: int = 42) -> None:
        self.method = method
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return apply_smote_tomek(
            X, y, method=self.method, random_state=self.random_state,
        )

    # imblearn Pipeline'ın param-grid mekanizması için
    def get_params(self, deep: bool = True) -> dict:
        return {"method": self.method, "random_state": self.random_state}

    def set_params(self, **params) -> "SafeResampler":
        for k, v in params.items():
            setattr(self, k, v)
        return self


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """R², RMSE, RPD, MAPE."""
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rpd = float(np.std(y_true) / rmse) if rmse > 0 else 0.0
    nonzero = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        if np.sum(nonzero) > 0
        else 0.0
    )
    return {"R2": r2, "RMSE": rmse, "RPD": rpd, "MAPE": mape}


def classify_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Accuracy / balanced acc / macro-precision/recall/F1 + confusion matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if class_names is not None and len(class_names) != len(labels):
        try:
            class_names = [class_names[int(l)] for l in labels if int(l) < len(class_names)]
            if len(class_names) != len(labels):
                class_names = None
        except Exception:
            class_names = None

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, zero_division=0
    )
    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "labels": [int(l) for l in labels],
        "class_names": class_names,
    }


def pheur_pass_fail(y_flav: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """Flavonoid PASS/FAIL eşiği: ``y_flav >= threshold`` → 1 (PASS), aksi 0 (FAIL).

    NOT: 'pheur' tarihsel değişken/fonksiyon adıdır; ≈%3.5 eşiği EMA değerlendirme
    raporundan (EMA/HMPC/464682/2016) gelir — Ph.Eur. kalite şartı DEĞİLDİR.
    """
    return (np.asarray(y_flav) >= threshold).astype(int)


def plot_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str = "",
    xlabel: str = "Gerçek",
    ylabel: str = "Tahmin",
) -> None:
    """y_true vs y_pred scatter + ideal çizgi + polinom trend + R²/RMSE/RPD kutusu."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5, s=40)
    lims = [
        min(y_true.min(), y_pred.min()) * 0.9,
        max(y_true.max(), y_pred.max()) * 1.1,
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="y = x")

    order = np.argsort(y_true)
    xs, ys = np.asarray(y_true)[order], np.asarray(y_pred)[order]
    if len(xs) >= 4 and np.ptp(xs) > 0:
        coeffs = np.polyfit(xs, ys, deg=2)
        grid = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(grid, np.polyval(coeffs, grid), color="darkgreen", linewidth=1.8,
                label="Trend (poly-2)")

    m = regression_metrics(y_true, y_pred)
    ax.text(0.05, 0.92,
            f"R² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.3f}\nRPD = {m['RPD']:.2f}",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, title: str = "") -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
    axes[0].axhline(0, color="r", linestyle="--")
    axes[0].set_xlabel("Tahmin")
    axes[0].set_ylabel("Kalıntı")
    axes[0].set_title("Kalıntı vs tahmin")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=20, color="steelblue", edgecolor="k")
    axes[1].set_xlabel("Kalıntı")
    axes[1].set_ylabel("Frekans")
    axes[1].set_title("Kalıntı dağılımı")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(
    cm: np.ndarray | list[list[int]],
    save_path: Path,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
) -> None:
    cm = np.asarray(cm)
    n = cm.shape[0]
    if class_names is None:
        class_names = [f"Sınıf {i}" for i in range(n)]
    elif len(class_names) != n:
        class_names = list(class_names)[:n]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    save_path: Path,
    top_n: int = 20,
    title: str = "Özellik önemleri",
) -> None:
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.barh(range(len(top_values)), top_values[::-1], color="steelblue", edgecolor="k")
    ax.set_yticks(range(len(top_values)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Önem")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(
    history: dict[str, list[float]],
    save_path: Path,
    title: str = "Öğrenme eğrisi",
) -> None:
    """Deep learning eğitim/doğrulama loss eğrisi.

    ``history`` dict'inde ``train_loss`` zorunlu, ``val_loss`` opsiyonel.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="train_loss", color="steelblue")
    if "val_loss" in history and history["val_loss"]:
        ax.plot(epochs, history["val_loss"], label="val_loss", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_json(metrics: dict, save_path: Path) -> None:
    """Metrik dict'ini JSON olarak yaz (numpy tipleri düzeltilir)."""

    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(
        json.dumps(_convert(metrics), indent=2, ensure_ascii=False), encoding="utf-8"
    )
