"""02_eda aşaması: dataset üzerinde temel keşifsel analiz.

``outputs/02_eda/`` altına şunları yazar:
  - ``distributions.png``       — Chl/Flav/NBI/stres histogramları
  - ``correlation_heatmap.png`` — özellikler arası korelasyon (top 30)
  - ``stats.csv``               — özellik bazlı temel istatistikler
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.core import paths
from src.core.logging_setup import get as get_logger
from src.m05_dataset import builder as dataset_builder

log = get_logger("m06_models.eda")


def _plot_target_distributions(data: dict, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, key, label in zip(
        axes.ravel(),
        ["y_chl", "y_flav", "y_nbi", "y_stress"],
        ["Klorofil", "Flavonol", "NBI", "Stres sınıfı"],
    ):
        y = data[key]
        finite = y[np.isfinite(y)]
        if key == "y_stress":
            counts = np.bincount(y.astype(int), minlength=4)
            ax.bar(["0\nsağlıklı", "1\nFD", "2\nbiyotik", "3\nabiyotik"], counts,
                   color="steelblue", edgecolor="k")
            for i, c in enumerate(counts):
                ax.text(i, c, str(c), ha="center", va="bottom")
        else:
            ax.hist(finite, bins=25, color="steelblue", edgecolor="k", alpha=0.85)
        ax.set_title(label, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Hedef değişken dağılımları", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation_heatmap(
    X: np.ndarray, feature_names: list[str], save_path: Path, top_n: int = 30
) -> None:
    df = pd.DataFrame(X, columns=feature_names)
    variances = df.var().sort_values(ascending=False)
    top_cols = variances.head(top_n).index.tolist()
    corr = df[top_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.8}, ax=ax, annot=False)
    ax.set_title(f"En yüksek varyanslı {top_n} özelliğin korelasyonu",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_stats(X: np.ndarray, feature_names: list[str], save_path: Path) -> None:
    df = pd.DataFrame(X, columns=feature_names)
    stats = df.describe(percentiles=[0.25, 0.5, 0.75]).T
    stats.to_csv(save_path, encoding="utf-8")


def run(cfg) -> None:
    """02_eda aşamasını çalıştır."""
    t0 = time.time()
    out_dir = paths.stage_dir("02_eda")

    data = dataset_builder.load()
    _plot_target_distributions(data, out_dir / "distributions.png")
    _plot_correlation_heatmap(data["X"], data["feature_names"],
                              out_dir / "correlation_heatmap.png")
    _save_stats(data["X"], data["feature_names"], out_dir / "stats.csv")

    paths.write_source_marker(out_dir, producer="src/m06_models/eda.py",
                              config_source=cfg.source)
    log.info("02_eda tamamlandı, süre=%.1fs", time.time() - t0)
