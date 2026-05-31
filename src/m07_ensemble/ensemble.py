"""Ensemble stage: tüm aşamaların metriklerini toplayıp özet rapor üretir.

Tüm pipeline çalıştıktan sonra şu çıktılar üretilir (``outputs/11_ensemble/`` altına):
    - ``final_report.csv``                 → regresyon (klasik + DL)
    - ``final_report_classification.csv``  → sınıflandırma (klasik + DL)
    - ``final_report_all.csv``             → birleşik (kaynak/aşama etiketli)
    - ``final_report_extras.csv``          → ordinal flav + anomaly flav (özet)
    - ``final_report_feature_selection.csv`` → SHAP top-N + RFE karşılaştırması özeti
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.core import paths
from src.core.logging_setup import get as get_logger

log = get_logger("m07_ensemble.ensemble")

REG_KEYS = ["R2", "RMSE", "RPD", "MAPE"]
CLS_KEYS = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"]


def _collect_metrics(stage_dir: Path, stage_label: str) -> list[dict[str, Any]]:
    """``<stage_dir>/<model>/<target>/metrics.json`` dosyalarını topla."""
    out: list[dict[str, Any]] = []
    if not stage_dir.exists():
        return out
    for model_dir in sorted(stage_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for target_dir in sorted(model_dir.iterdir()):
            if not target_dir.is_dir():
                continue
            mfile = target_dir / "metrics.json"
            if not mfile.exists():
                continue
            try:
                data = json.loads(mfile.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("metrics.json okunamadı (%s): %s", mfile, exc)
                continue
            out.append({
                "stage": stage_label,
                "model": model_dir.name,
                "target": target_dir.name,
                **data,
            })
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _split_reg_cls(rows: list[dict[str, Any]]):
    reg, cls = [], []
    for r in rows:
        if "R2" in r:
            reg.append(r)
        elif "accuracy" in r:
            cls.append(r)
    return reg, cls


def _read_csv_auto(path: Path) -> pd.DataFrame:
    """Excel'in TR yerel ayarda kaydettiği ``;`` ayraçlı CSV'leri de tanır."""
    return pd.read_csv(path, sep=None, engine="python")


def _collect_ordinal(_unused: Path | None = None) -> list[dict[str, Any]]:
    """``ordinal_flavonol_report.csv``'den mean satırlarını al.

    Numaralı aşama dizininden okur (eski reports/ kaldırıldı).
    """
    csv_path = paths.OUTPUTS_DIR / "09_ordinal_flavonol" / "ordinal_flavonol_report.csv"
    if not csv_path.exists():
        return []
    df = _read_csv_auto(csv_path)
    df = df[df["fold"].astype(str) == "mean"]
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "stage": "ordinal",
            "model": str(r.get("model", "")),
            "target": f"flavonol_{r.get('scheme', 'ord')}",
            "accuracy": float(r["accuracy"]),
            "balanced_accuracy": float(r["balanced_accuracy"]),
            "macro_f1": float(r["macro_f1"]),
            "mae": float(r.get("mae", float("nan"))) if "mae" in r else None,
            "qwk": float(r.get("qwk", float("nan"))) if "qwk" in r else None,
        })
    return rows


def _collect_anomaly(_unused: Path | None = None) -> list[dict[str, Any]]:
    """``anomaly_flavonol_report.csv``'den dedektör satırlarını al.

    Numaralı aşama dizininden okur (eski reports/ kaldırıldı).
    """
    csv_path = paths.OUTPUTS_DIR / "10_anomaly_flavonol" / "anomaly_flavonol_report.csv"
    if not csv_path.exists():
        return []
    df = _read_csv_auto(csv_path)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "stage": "anomaly",
            "model": str(r.get("detector", "")),
            "target": "flavonol_pheur",
            "precision": float(r.get("precision", float("nan"))),
            "recall": float(r.get("recall", float("nan"))),
            "f1": float(r.get("f1", float("nan"))),
            "auc_roc": float(r.get("auc_roc", float("nan"))),
            "threshold": (float(r["threshold"])
                          if "threshold" in r and pd.notna(r.get("threshold", None))
                          else None),
        })
    return rows


def _collect_feature_selection(_unused: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """SHAP top özellikleri + RFE karşılaştırma satırlarını topla.

    Numaralı aşama dizinlerinden okur (eski reports/ kaldırıldı).
    """
    shap_rows: list[dict[str, Any]] = []
    shap_csv = paths.OUTPUTS_DIR / "04_feature_shap" / "shap_top_features.csv"
    if shap_csv.exists():
        df = _read_csv_auto(shap_csv).head(20)
        for _, r in df.iterrows():
            shap_rows.append({
                "method": "shap",
                "rank": int(r["rank"]),
                "feature": str(r["feature"]),
                "score": float(r["mean_abs_shap"]),
            })

    rfe_rows: list[dict[str, Any]] = []
    rfe_csv = paths.OUTPUTS_DIR / "05_feature_rfe" / "rfe_model_comparison.csv"
    if rfe_csv.exists():
        df = _read_csv_auto(rfe_csv)
        for _, r in df.iterrows():
            rfe_rows.append({
                "method": "rfe",
                "model": str(r["model"]),
                "n_features_full": int(r["n_features_full"]),
                "macro_f1_full": float(r["macro_f1_full"]),
                "n_features_rfe": int(r["n_features_rfe"]),
                "macro_f1_rfe": float(r["macro_f1_rfe"]),
                "delta_f1": float(r["delta_f1"]),
            })
    return shap_rows, rfe_rows


def run(cfg) -> None:
    tdir = paths.stage_dir("11_ensemble")
    # reports_dir kaldırıldı; tüketici fonksiyonlar artık numaralı dizinden okur.

    # Klasik regresyon/sınıflandırma + DL
    classical_reg = _collect_metrics(paths.OUTPUTS_DIR / "06_regression", "classical")
    classical_cls = _collect_metrics(paths.OUTPUTS_DIR / "07_classification", "classical")
    deep_all = _collect_metrics(paths.OUTPUTS_DIR / "08_deep_learning", "deep_learning")
    deep_reg, deep_cls = _split_reg_cls(deep_all)

    reg_rows = classical_reg + deep_reg
    cls_rows = classical_cls + deep_cls

    # Ekstra modüller (ordinal flavonol + anomaly flavonol)
    ordinal_rows = _collect_ordinal()
    anomaly_rows = _collect_anomaly()

    # ---- Regresyon raporu (klasik + DL birlikte) ----
    reg_columns = ["stage", "model", "target", *REG_KEYS]
    _write_csv(reg_rows, tdir / "final_report.csv", reg_columns)

    # ---- Sınıflandırma raporu (klasik + DL + ordinal) ----
    cls_columns = ["stage", "model", "target", *CLS_KEYS, "mae", "qwk"]
    _write_csv(cls_rows + ordinal_rows, tdir / "final_report_classification.csv", cls_columns)

    # ---- Birleşik özet (her şey) ----
    all_rows = reg_rows + cls_rows + ordinal_rows + anomaly_rows
    all_columns = [
        "stage", "model", "target",
        *REG_KEYS, *CLS_KEYS,
        "mae", "qwk",
        "precision", "recall", "f1", "auc_roc", "threshold",
    ]
    _write_csv(all_rows, tdir / "final_report_all.csv", all_columns)

    # ---- Ekstralar (ordinal + anomaly özet) ----
    extras_columns = [
        "stage", "model", "target",
        "accuracy", "balanced_accuracy", "macro_f1", "mae", "qwk",
        "precision", "recall", "f1", "auc_roc", "threshold",
    ]
    _write_csv(ordinal_rows + anomaly_rows, tdir / "final_report_extras.csv", extras_columns)

    # ---- Feature selection özeti (SHAP top-20 + RFE karşılaştırma) ----
    shap_rows, rfe_rows = _collect_feature_selection()
    fs_path = tdir / "final_report_feature_selection.csv"
    if shap_rows or rfe_rows:
        with fs_path.open("w", encoding="utf-8", newline="") as f:
            f.write("# SHAP top-20\n")
            if shap_rows:
                writer = csv.DictWriter(f, fieldnames=["method", "rank", "feature", "score"])
                writer.writeheader()
                for r in shap_rows:
                    writer.writerow(r)
            f.write("\n# RFE comparison (baseline vs selected)\n")
            if rfe_rows:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["method", "model", "n_features_full", "macro_f1_full",
                                "n_features_rfe", "macro_f1_rfe", "delta_f1"],
                )
                writer.writeheader()
                for r in rfe_rows:
                    writer.writerow(r)

    paths.write_source_marker(tdir, producer="src/m07_ensemble/ensemble.py", config_source=cfg.source)
    log.info(
        "11_ensemble tamamlandı: regresyon=%d (klasik=%d, DL=%d) | "
        "sınıflandırma=%d (klasik=%d, DL=%d) | ordinal=%d | anomaly=%d | "
        "SHAP=%d | RFE=%d",
        len(reg_rows), len(classical_reg), len(deep_reg),
        len(cls_rows), len(classical_cls), len(deep_cls),
        len(ordinal_rows), len(anomaly_rows),
        len(shap_rows), len(rfe_rows),
    )
