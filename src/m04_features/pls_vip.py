"""PLS-VIP (Variable Importance in Projection) skoru.

VIP, PLS regresyonun her özelliğinin yanıt değişkenini (y) açıklamadaki
bağıl katkısını ölçer. Geleneksel eşik **VIP > 1.0** "önemli" sayılır.

Formül (Wold 1995, Chong & Jun 2005):

    VIP_j = sqrt( p * sum_a [ SS_a * w_aj^2 / |w_a|^2 ] / SS_total )

burada
    p      : özellik sayısı
    SS_a   : a'ıncı PLS bileşeninin y-açıklayıcılığı (tek-y için: q_a^2 * |t_a|^2)
    w_aj   : j'inci özelliğin a'ıncı bileşendeki ağırlığı
    SS_tot : toplam y-açıklayıcılık

Pipeline çıktıları (``outputs/05b_pls_vip/``):
    - vip_<target>.csv       — feature_index, feature, VIP (azalan sıralı)
    - vip_<target>.png       — top-30 bar grafik
    - vip_summary.txt        — Türkçe özet (her hedef için top-15 + toplam>1 sayısı)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from src.core import paths
from src.core.logging_setup import get as get_logger

log = get_logger("m04_features.pls_vip")

_TARGETS = (("chlorophyll", "y_chl.npy"),
            ("flavonol", "y_flav.npy"),
            ("nbi", "y_nbi.npy"))


def compute_vip(pls: PLSRegression) -> np.ndarray:
    """Eğitilmiş PLSRegression için VIP skorlarını döndür (shape (p,))."""
    t = pls.x_scores_           # (n, A) — skor matrisi
    w = pls.x_weights_          # (p, A) — ağırlık matrisi
    q = pls.y_loadings_         # (k, A) — y-yükleri (k=hedef sayısı=1)
    p, A = w.shape
    # Bileşen başına y-açıklama: |t_a|^2 * |q_a|^2  (tek-y → skaler q_a)
    ss_per_comp = np.sum(t**2, axis=0) * np.sum(q**2, axis=0)
    total_ss = float(np.sum(ss_per_comp))
    if total_ss <= 0:
        return np.zeros(p)
    w_norm_sq = np.sum(w**2, axis=0)  # (A,)
    w_norm_sq = np.where(w_norm_sq > 0, w_norm_sq, 1.0)
    weighted = (w**2) * (ss_per_comp / w_norm_sq)[None, :]  # (p, A)
    vip = np.sqrt(p * weighted.sum(axis=1) / total_ss)
    return vip


def _fit_pls(X: np.ndarray, y: np.ndarray, n_components: int) -> PLSRegression:
    n_comp = max(1, min(n_components, X.shape[1], X.shape[0] - 1))
    pls = PLSRegression(n_components=n_comp, scale=False, max_iter=500)
    pls.fit(X, y)
    return pls


def _plot_top_n(
    vip: np.ndarray, feature_names: list[str], target: str,
    save_path: Path, n_top: int = 30,
) -> None:
    order = np.argsort(vip)[::-1][:n_top]
    fig, ax = plt.subplots(1, 1, figsize=(9, 0.28 * n_top + 1.5))
    y_pos = np.arange(len(order))[::-1]
    ax.barh(y_pos, vip[order], color="steelblue")
    ax.axvline(1.0, color="indianred", linestyle="--", linewidth=1, label="VIP=1")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8)
    ax.set_xlabel("VIP skoru")
    ax.set_title(f"PLS-VIP top-{n_top} | target={target}",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(cfg=None) -> Path:
    """01_dataset üzerinden her regresyon hedefi için PLS-VIP üret."""
    n_components = 12
    if cfg is not None:
        n_components = int(cfg.get("models.regression.pls.n_components", n_components))

    ds_dir = paths.OUTPUTS_DIR / "01_dataset"
    X = np.load(ds_dir / "X.npy")
    feature_names = json.loads(
        (ds_dir / "feature_names.json").read_text(encoding="utf-8")
    )
    log.info("PLS-VIP: X=%s, n_components=%d", X.shape, n_components)

    # Tek kaynak: numaralı aşama dizini (eski reports/pls_vip kaldırıldı).
    stage_dir = paths.stage_dir("05b_pls_vip")
    out_dir = stage_dir

    # Standardize (X) — y ham bırakılır; sklearn PLS scale=True ile zaten merkezler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(np.where(np.isfinite(X), X, 0.0))

    summary: list[str] = ["PLS-VIP Özeti", "=" * 50]

    for target, fname in _TARGETS:
        path = ds_dir / fname
        if not path.exists():
            log.warning("Hedef yok, atlanıyor: %s", fname)
            continue
        y = np.load(path)
        valid = np.isfinite(y)
        if valid.sum() < 10:
            log.warning("Hedef %s için yeterli geçerli örnek yok (n=%d), atlanıyor",
                        target, int(valid.sum()))
            continue

        pls = _fit_pls(Xs[valid], y[valid].reshape(-1, 1), n_components)
        vip = compute_vip(pls)

        df = pd.DataFrame({
            "feature_index": np.arange(len(vip)),
            "feature": feature_names,
            "vip": vip,
        }).sort_values("vip", ascending=False).reset_index(drop=True)

        df.to_csv(out_dir / f"vip_{target}.csv", index=False, encoding="utf-8")
        _plot_top_n(vip, feature_names, target, out_dir / f"vip_{target}.png")

        n_above = int(np.sum(vip > 1.0))
        log.info("VIP %s: n_above_1=%d/%d, top-3=%s",
                 target, n_above, len(vip),
                 [(feature_names[i], round(vip[i], 2))
                  for i in np.argsort(vip)[::-1][:3]])
        summary.append(f"\n[{target}] VIP>1: {n_above}/{len(vip)}")
        for i in np.argsort(vip)[::-1][:15]:
            summary.append(f"  {feature_names[i]:<28} VIP={vip[i]:.3f}")

    summary_text = "\n".join(summary) + "\n"
    (out_dir / "vip_summary.txt").write_text(summary_text, encoding="utf-8-sig")

    paths.write_source_marker(
        stage_dir,
        producer="src/m04_features/pls_vip.py",
        config_source=cfg.source if cfg is not None else None,
    )
    log.info("PLS-VIP raporu yazıldı: %s", out_dir)
    return out_dir
