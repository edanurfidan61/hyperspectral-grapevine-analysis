"""Birleşik özellik önem raporu — SHAP + RFE + GA + PLS-VIP.

Dört farklı yöntemin sıralamalarını "ortalama rank" (Borda benzeri) ile
birleştirip tek bir top-N tablosu üretir. Tezde "hangi dalga boyları en bilgili"
sorusunu tek grafikte raporlamak için tasarlanmıştır.

Birleştirme kuralı (Borda count):
    - Her kaynak için feature'lar sıralanır (en yüksek skor → rank 1).
    - Bir feature kaynak X'te yoksa, rank = N+1 (penalty).
    - Final skor = ``mean(rank)`` (küçük = önemli).

Çıktılar (``outputs/13b_feature_consensus/``):
    - feature_consensus_all.csv      — tüm feature'lar, her kaynaktan rank + ort.
    - feature_consensus_top.csv      — top-N (default 50)
    - feature_consensus_top.png      — top-N bar grafik (ortalama rank)
    - wavelength_top.csv             — sadece spektral bant feature'ları (snv/d1)
                                       wavelength bilgisiyle, top-N
    - consensus_summary.txt          — Türkçe özet
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from src.core import paths
from src.core.logging_setup import get as get_logger

log = get_logger("m04_features.feature_consensus")

# Spektral bant feature adlarından wavelength çıkarmak için regex
# Eşleştirir: snv_R670, d1snv_R670, R670_mean, NDVI_mean, vb.
_WL_RE = re.compile(r"R(\d{3,4})", re.IGNORECASE)


def _rank_from_csv(
    csv_path: Path, feature_col: str, score_col: str, ascending: bool = False,
) -> dict[str, int]:
    """Bir CSV'den {feature_name: rank} dict'i çıkar (rank=1 en önemli)."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if feature_col not in df.columns or score_col not in df.columns:
        return {}
    df = df[[feature_col, score_col]].dropna()
    df = df.sort_values(score_col, ascending=ascending).reset_index(drop=True)
    return {str(row[feature_col]): i + 1 for i, row in df.iterrows()}


def _collect_rankings() -> dict[str, dict[str, int]]:
    """Mevcut tüm kaynaklardan ranking'leri topla.

    Kaynaklar artık doğrudan numaralı aşama dizinlerinden okunur
    (eski outputs/reports/ hub kaldırıldı).
    """
    out: dict[str, dict[str, int]] = {}

    # SHAP — outputs/04_feature_shap/shap_top_features.csv
    shap_csv = paths.OUTPUTS_DIR / "04_feature_shap" / "shap_top_features.csv"
    r = _rank_from_csv(shap_csv, "feature", "mean_abs_shap", ascending=False)
    if r:
        out["shap"] = r
        log.info("SHAP ranking: %d feature", len(r))

    # RFE — outputs/05_feature_rfe/rfe_selected_features.csv (rank kolonu var)
    rfe_csv = paths.OUTPUTS_DIR / "05_feature_rfe" / "rfe_selected_features.csv"
    if rfe_csv.exists():
        df = pd.read_csv(rfe_csv)
        if "feature" in df.columns and "rank" in df.columns:
            out["rfe"] = {str(f): int(r) for f, r in zip(df["feature"], df["rank"])}
            log.info("RFE ranking: %d feature", len(out["rfe"]))

    # GA — outputs/12_ga_feature_selection/<target>_<model>/ga_best_features.txt
    ga_root = paths.OUTPUTS_DIR / "12_ga_feature_selection"
    if ga_root.exists():
        for ga_dir in ga_root.glob("*"):
            # _checkpoint gibi yardımcı klasörleri atla
            if not ga_dir.is_dir() or ga_dir.name.startswith("_"):
                continue
            feat_txt = ga_dir / "ga_best_features.txt"
            if not feat_txt.exists():
                continue
            names = [l.strip() for l in feat_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
            if names:
                key = f"ga_{ga_dir.name}"
                out[key] = {n: i + 1 for i, n in enumerate(names)}
                log.info("GA[%s] ranking: %d feature", ga_dir.name, len(names))

    # PLS-VIP — outputs/05b_pls_vip/vip_<target>.csv (her hedef ayrı dosya)
    pls_dir = paths.OUTPUTS_DIR / "05b_pls_vip"
    if pls_dir.exists():
        for vip_csv in pls_dir.glob("vip_*.csv"):
            target = vip_csv.stem.replace("vip_", "")
            df = pd.read_csv(vip_csv)
            if "feature" in df.columns and "vip" in df.columns:
                df = df.sort_values("vip", ascending=False).reset_index(drop=True)
                out[f"vip_{target}"] = {
                    str(row["feature"]): i + 1 for i, row in df.iterrows()
                }
                log.info("VIP[%s] ranking: %d feature", target, len(out[f"vip_{target}"]))

    return out


def _extract_wavelength(name: str) -> int | None:
    m = _WL_RE.search(name)
    return int(m.group(1)) if m else None


def _is_spectral_feature(name: str) -> bool:
    return name.lower().startswith(("snv_r", "d1snv_r"))


def run(cfg=None, top_n: int = 50) -> Path:
    """Tüm raporları tara, consensus üret."""
    if cfg is not None:
        top_n = int(cfg.get("consensus.top_n", top_n))

    # Tek kaynak: numaralı aşama dizini (eski reports/consensus kaldırıldı).
    stage_dir = paths.stage_dir("13b_feature_consensus")
    out_dir = stage_dir

    # Feature isimlerini dataset'ten al (tüm evren)
    fn_path = paths.OUTPUTS_DIR / "01_dataset" / "feature_names.json"
    all_features: list[str] = (
        json.loads(fn_path.read_text(encoding="utf-8"))
        if fn_path.exists() else []
    )

    rankings = _collect_rankings()
    if not rankings:
        log.warning("Hiçbir ranking kaynağı bulunamadı; consensus atlanıyor")
        return out_dir
    log.info("Toplam %d kaynak bulundu: %s", len(rankings), list(rankings))

    # Tüm feature evrenini garanti et (ranking'lerde olmayanlar da dahil)
    feature_universe: set[str] = set(all_features)
    for src in rankings.values():
        feature_universe.update(src.keys())
    feature_list = sorted(feature_universe)

    # Penalty: kaynaktaki maksimum rank + 1 (yani "yok" demek "en kötü"den daha kötü)
    rows = []
    for f in feature_list:
        ranks_per_source: dict[str, int] = {}
        for src_name, ranks in rankings.items():
            penalty = max(ranks.values()) + 1
            ranks_per_source[src_name] = ranks.get(f, penalty)
        mean_rank = float(np.mean(list(ranks_per_source.values())))
        median_rank = float(np.median(list(ranks_per_source.values())))
        appearances = sum(1 for src in rankings.values() if f in src)
        wl = _extract_wavelength(f)
        rows.append({
            "feature": f,
            "wavelength_nm": wl,
            "mean_rank": mean_rank,
            "median_rank": median_rank,
            "appearances": appearances,
            **{f"rank_{k}": v for k, v in ranks_per_source.items()},
        })

    df = pd.DataFrame(rows).sort_values(
        ["mean_rank", "median_rank"], ascending=True,
    ).reset_index(drop=True)

    # ---- CSV çıktıları ----
    df.to_csv(out_dir / "feature_consensus_all.csv", index=False, encoding="utf-8")

    top_df = df.head(top_n)
    top_df.to_csv(out_dir / "feature_consensus_top.csv", index=False, encoding="utf-8")

    # ---- Wavelength tablosu (sadece spektral bant feature'ları) ----
    spectral = df[df["feature"].apply(_is_spectral_feature)].copy()
    spectral = spectral.dropna(subset=["wavelength_nm"])
    spectral["wavelength_nm"] = spectral["wavelength_nm"].astype(int)
    wl_top = spectral.head(top_n)
    wl_top.to_csv(out_dir / "wavelength_top.csv", index=False, encoding="utf-8")

    # ---- Bar grafik (top-N ortalama rank, ters çevrilmiş = küçük en iyi) ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 0.3 * top_n + 1.5))
    y_pos = np.arange(len(top_df))[::-1]
    ax.barh(y_pos, top_df["mean_rank"], color="steelblue", edgecolor="k")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df["feature"], fontsize=7)
    ax.set_xlabel("Ortalama rank (küçük = daha önemli)")
    ax.set_title(f"Birleşik özellik önemi top-{top_n} "
                 f"({len(rankings)} kaynak: SHAP+RFE+GA+VIP)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "feature_consensus_top.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Türkçe özet ----
    summary = [
        "Birleşik Özellik Önem Özeti (Consensus)",
        "=" * 60,
        f"Kaynak sayısı       : {len(rankings)}",
        f"Kaynaklar           : {', '.join(rankings.keys())}",
        f"Toplam feature      : {len(df)}",
        f"Top-N gösterilen    : {top_n}",
        "",
        f"En önemli {min(20, top_n)} özellik (ortalama rank ile):",
        "-" * 60,
    ]
    for _, r in df.head(20).iterrows():
        wl_str = f" [{int(r['wavelength_nm'])} nm]" if pd.notna(r["wavelength_nm"]) else ""
        summary.append(
            f"  {r['feature']:<28}{wl_str:<11} ort.rank={r['mean_rank']:5.1f}  "
            f"(görünüm: {int(r['appearances'])}/{len(rankings)})"
        )
    summary += [
        "-" * 60,
        "",
        f"En önemli {min(20, top_n)} dalga boyu (sadece snv/d1 spektral bantlar):",
        "-" * 60,
    ]
    for _, r in spectral.head(20).iterrows():
        summary.append(
            f"  {int(r['wavelength_nm'])} nm  ({r['feature']:<28}) "
            f"ort.rank={r['mean_rank']:5.1f}"
        )
    summary_text = "\n".join(summary) + "\n"
    (out_dir / "consensus_summary.txt").write_text(summary_text, encoding="utf-8-sig")

    log.info("Consensus yazıldı: %s", out_dir)

    paths.write_source_marker(
        stage_dir,
        producer="src/m04_features/feature_consensus.py",
        config_source=cfg.source if cfg is not None else None,
    )
    return out_dir
