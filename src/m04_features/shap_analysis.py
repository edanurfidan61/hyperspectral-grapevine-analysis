"""SHAP tabanlı özellik önemi analizi (LightGBM + stres etiketleri).

Çıktılar:
    outputs/04_feature_shap/shap_summary_bar.png
    outputs/04_feature_shap/shap_summary_beeswarm.png
    outputs/04_feature_shap/shap_top_features.csv
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from src.core import paths
from src.core.logging_setup import get as get_logger

log = get_logger("m04_features.shap_analysis")

TOP_N = 30


def _load_data():
    ds = paths.OUTPUTS_DIR / "01_dataset"
    X = np.load(ds / "X.npy")
    y_stress = np.load(ds / "y_stress.npy")
    feature_names = json.loads((ds / "feature_names.json").read_text(encoding="utf-8"))
    return X, y_stress, feature_names


def _build_lgbm(random_state: int):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1, verbosity=-1, force_col_wise=True,
        ), "LightGBM"
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        log.warning("LightGBM yok — RF fallback")
        return RandomForestClassifier(
            n_estimators=500, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        ), "RF"


def run(cfg=None) -> Path:
    """SHAP analizi: LightGBM'i tüm dataset üzerinde fit et, TreeExplainer ile yorumla."""
    import shap

    random_state = int(cfg.get("models.random_state", 42)) if cfg is not None else 42

    X, y_stress, feature_names = _load_data()
    log.info("SHAP: X=%s, n_class=%d", X.shape, len(np.unique(y_stress)))

    model, label = _build_lgbm(random_state)
    model.fit(X, y_stress)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # multi-class için: shap_values list[K] (eski) veya array (n,f,K) (yeni). Hepsini normalize et.
    if isinstance(shap_values, list):
        sv_stack = np.stack([np.abs(s) for s in shap_values], axis=-1)  # (n,f,K)
    else:
        sv = np.asarray(shap_values)
        sv_stack = np.abs(sv) if sv.ndim == 3 else np.abs(sv)[..., None]

    # |SHAP| ortalama: önce sınıflar üzerinden, sonra örnekler üzerinden
    mean_abs_per_class = sv_stack.mean(axis=0)            # (f, K)
    mean_abs = mean_abs_per_class.mean(axis=-1)           # (f,)
    if mean_abs.shape[0] != X.shape[1]:
        mean_abs = mean_abs.reshape(-1)[:X.shape[1]]

    order = np.argsort(mean_abs)[::-1]
    ranking = pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "feature": [feature_names[i] for i in order],
        "mean_abs_shap": mean_abs[order],
    })

    # Tek kaynak: numaralı aşama dizini (eski reports/shap kaldırıldı).
    stage_dir = paths.stage_dir("04_feature_shap")
    out_dir = stage_dir
    csv_path = out_dir / "shap_top_features.csv"
    ranking.to_csv(csv_path, index=False, encoding="utf-8")
    log.info("Ranking yazıldı: %s", csv_path)

    # ---- bar plot (top 30) --------------------------------------------------
    top = ranking.head(TOP_N)[::-1]
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    ax.barh(range(len(top)), top["mean_abs_shap"], color="steelblue", edgecolor="k")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP özellik önemi (top {TOP_N}, model={label})",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    bar_path = out_dir / "shap_summary_bar.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Bar plot yazıldı: %s", bar_path)

    # ---- beeswarm (top 30, sınıf-ortalı SHAP) -------------------------------
    # multi-class için işaret bilgisi sınıf bazında — sınıflar üstünde mean alıyoruz
    if isinstance(shap_values, list):
        sv_signed = np.mean(np.stack(shap_values, axis=-1), axis=-1)  # (n,f)
    else:
        sv = np.asarray(shap_values)
        sv_signed = sv.mean(axis=-1) if sv.ndim == 3 else sv

    top_idx = order[:TOP_N]
    try:
        bee_path = out_dir / "shap_summary_beeswarm.png"
        plt.figure(figsize=(9, 8))
        shap.summary_plot(
            sv_signed[:, top_idx],
            X[:, top_idx],
            feature_names=[feature_names[i] for i in top_idx],
            show=False,
            max_display=TOP_N,
        )
        plt.tight_layout()
        plt.savefig(bee_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        log.info("Beeswarm plot yazıldı: %s", bee_path)
    except Exception as exc:
        log.warning("Beeswarm üretilemedi: %s", exc)

    # ---- konsola top 20 ----------------------------------------------------
    log.info("=== TOP 20 SHAP özellikleri ===")
    for _, row in ranking.head(20).iterrows():
        log.info("  %2d. %-40s |SHAP|=%.5f",
                 int(row["rank"]), row["feature"][:40], row["mean_abs_shap"])

    paths.write_source_marker(
        stage_dir,
        producer="src/m04_features/shap_analysis.py",
        config_source=cfg.source if cfg is not None else None,
    )
    return csv_path


if __name__ == "__main__":
    run()
