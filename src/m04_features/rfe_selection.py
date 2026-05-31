"""Recursive Feature Elimination (RFECV) ile özellik seçimi.

- RFECV (RandomForest tabanlı) ile optimum özellik sayısını bul
- Seçilen özelliklerle LightGBM ve RF'yi yeniden eğit, baseline (tüm özellikler) ile karşılaştır
- ``X_rfe.npy`` filtreli matrisi ``outputs/01_dataset/`` altına kaydet (sonraki aşamalar kullanabilir)

Çıktılar:
    outputs/05_feature_rfe/rfe_cv_scores.png
    outputs/05_feature_rfe/rfe_selected_features.csv
    outputs/05_feature_rfe/rfe_model_comparison.csv
    outputs/01_dataset/X_rfe.npy + outputs/01_dataset/rfe_feature_indices.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from src.core import paths
from src.core.cv import load_groups, make_cv_splitter
from src.core.logging_setup import get as get_logger

log = get_logger("m04_features.rfe_selection")


def _load_data(group_key: str = "leaf"):
    ds = paths.OUTPUTS_DIR / "01_dataset"
    X = np.load(ds / "X.npy")
    y = np.load(ds / "y_stress.npy")
    feature_names = json.loads((ds / "feature_names.json").read_text(encoding="utf-8"))
    groups = load_groups(ds, key=group_key)
    return X, y, feature_names, groups


def _build_lgbm(random_state: int):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=500, learning_rate=0.05,
            class_weight="balanced", random_state=random_state,
            verbose=-1, verbosity=-1, force_col_wise=True,
        ), "lightgbm"
    except Exception:
        return RandomForestClassifier(
            n_estimators=500, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        ), "rf_fallback"


def _cv_macro_f1(estimator, X, y, cv, groups=None) -> tuple[float, float]:
    if groups is not None:
        scores = cross_val_score(
            estimator, X, y, scoring="f1_macro", cv=cv, n_jobs=-1, groups=groups,
        )
    else:
        scores = cross_val_score(estimator, X, y, scoring="f1_macro", cv=cv, n_jobs=-1)
    return float(np.mean(scores)), float(np.std(scores))


def _plot_cv_scores(rfecv: RFECV, n_features_total: int, save_path: Path) -> None:
    """RFECV n_features_to_select grid'i için ortalama±std F1 eğrisi."""
    cv_results = rfecv.cv_results_
    mean_scores = np.asarray(cv_results["mean_test_score"])
    std_scores = np.asarray(cv_results["std_test_score"])

    # cv_results sırası: en az min_features_to_select örnekten başlar, step kadar artar
    min_feat = int(getattr(rfecv, "min_features_to_select", 1))
    step = int(getattr(rfecv, "step", 1))
    x = min_feat + np.arange(len(mean_scores)) * step
    x = np.minimum(x, n_features_total)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(x, mean_scores, marker="o", color="steelblue", label="Mean CV macro-F1")
    ax.fill_between(x, mean_scores - std_scores, mean_scores + std_scores,
                    alpha=0.25, color="steelblue", label="±1 std")
    ax.axvline(rfecv.n_features_, color="indianred", linestyle="--",
               label=f"Optimum = {rfecv.n_features_}")
    ax.set_xlabel("Seçilen özellik sayısı")
    ax.set_ylabel("Macro-F1 (CV ortalaması)")
    ax.set_title("RFECV — özellik sayısı vs CV F1", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(cfg=None) -> Path:
    cv_splits = int(cfg.get("models.cv", 5)) if cfg is not None else 5
    random_state = int(cfg.get("models.random_state", 42)) if cfg is not None else 42
    group_key = str(cfg.get("cv.group_key", "leaf")) if cfg is not None else "leaf"

    X, y, feature_names, groups = _load_data(group_key=group_key)
    log.info("RFECV: X=%s, n_class=%d, groups=%s",
             X.shape, len(np.unique(y)),
             "yok" if groups is None else f"{group_key} ({len(set(groups.tolist()))} unique)")

    estimator = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )
    cv = make_cv_splitter(
        n_splits=cv_splits, task="classification",
        groups=groups, random_state=random_state,
    )

    log.info("RFECV başlıyor (step=10, scoring=f1_macro, cv=%d)...", cv_splits)
    rfecv = RFECV(
        estimator=estimator,
        step=10,
        cv=cv,
        scoring="f1_macro",
        min_features_to_select=10,
        n_jobs=-1,
    )
    if groups is not None:
        rfecv.fit(X, y, groups=groups)
    else:
        rfecv.fit(X, y)
    log.info("RFECV bitti: optimum n_features=%d (toplam %d)",
             rfecv.n_features_, X.shape[1])

    selected_mask = rfecv.support_
    selected_idx = np.where(selected_mask)[0]
    selected_names = [feature_names[i] for i in selected_idx]

    # Tek kaynak: numaralı aşama dizini (eski reports/rfe kaldırıldı).
    stage_dir = paths.stage_dir("05_feature_rfe")
    out_dir = stage_dir

    # 1) CV skor grafiği
    _plot_cv_scores(rfecv, X.shape[1], out_dir / "rfe_cv_scores.png")
    log.info("CV plot yazıldı")

    # 2) Seçilen özellik listesi
    sel_df = pd.DataFrame({
        "rank": np.arange(1, len(selected_names) + 1),
        "feature_index": selected_idx.tolist(),
        "feature": selected_names,
        "rfe_ranking": rfecv.ranking_[selected_idx].tolist(),
    })
    sel_df.to_csv(out_dir / "rfe_selected_features.csv", index=False, encoding="utf-8")

    # 3) Filtrelenmiş matris ve indeks haritası
    ds_dir = paths.OUTPUTS_DIR / "01_dataset"
    X_rfe = X[:, selected_mask]
    np.save(ds_dir / "X_rfe.npy", X_rfe)
    (ds_dir / "rfe_feature_indices.json").write_text(
        json.dumps({"selected_indices": selected_idx.tolist(),
                    "selected_features": selected_names},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("X_rfe kaydedildi: %s (shape=%s)", ds_dir / "X_rfe.npy", X_rfe.shape)

    # 4) Baseline vs RFE karşılaştırması
    log.info("Karşılaştırma: baseline (tüm özellikler) vs RFE seçimi")
    rows = []
    lgbm_full, lgbm_label = _build_lgbm(random_state)
    rf_full = RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )

    for model_name, est in (("lightgbm", lgbm_full), ("random_forest", rf_full)):
        # baseline (tüm özellikler)
        m_full, s_full = _cv_macro_f1(est, X, y, cv, groups=groups)
        # RFE altset
        m_rfe, s_rfe = _cv_macro_f1(est, X_rfe, y, cv, groups=groups)
        delta = m_rfe - m_full
        rows.append({
            "model": model_name,
            "n_features_full": int(X.shape[1]),
            "macro_f1_full": m_full,
            "std_f1_full": s_full,
            "n_features_rfe": int(X_rfe.shape[1]),
            "macro_f1_rfe": m_rfe,
            "std_f1_rfe": s_rfe,
            "delta_f1": delta,
        })
        log.info("  %s: full=%.3f±%.3f (n=%d) → rfe=%.3f±%.3f (n=%d) Δ=%+.3f",
                 model_name, m_full, s_full, X.shape[1],
                 m_rfe, s_rfe, X_rfe.shape[1], delta)

    cmp_df = pd.DataFrame(rows)
    cmp_csv = out_dir / "rfe_model_comparison.csv"
    cmp_df.to_csv(cmp_csv, index=False, encoding="utf-8")
    log.info("Karşılaştırma yazıldı: %s", cmp_csv)

    paths.write_source_marker(
        stage_dir,
        producer="src/m04_features/rfe_selection.py",
        config_source=cfg.source if cfg is not None else None,
    )
    return cmp_csv


if __name__ == "__main__":
    run()
