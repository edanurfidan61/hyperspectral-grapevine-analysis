"""Tüm modellerin metriklerini tek tabloda toplayan rapor üretici.

Pipeline'ın farklı aşamalarına dağılmış ``metrics.json`` ve ``*_report.csv``
dosyalarını tarayıp tek bir tabloya konsolide eder:

    Aşama | Hedef | Model | R² | Accuracy | BalancedAcc | MacroF1 |
    Precision | Recall | F1 | RMSE | RPD | MAPE | MAE | Notlar

Çıktılar (``outputs/14_model_summary/``):
    - all_models.csv      — UTF-8 BOM (Excel uyumlu)
    - all_models.xlsx     — openpyxl varsa; yoksa atlanır
    - summary.txt         — kısa Türkçe özet
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import paths
from src.core.logging_setup import get as get_logger

log = get_logger("m07_ensemble.model_summary")

# Aşama dizinleri ve görev tipleri
_REGRESSION_DIR = "06_regression"
_REGRESSION_TUNED_DIR = "06b_regression_tuned"          # GÖREV 2: coarse+fine HP tuning
_CLASSIFICATION_DIR = "07_classification"
_CLASSIFICATION_RESAMPLED_DIR = "07_classification_resampled"  # GÖREV 1: SMOTE+Tomek
_CLASSIFICATION_TUNED_DIR = "07b_classification_tuned"  # GÖREV 2: coarse+fine HP tuning
_DEEP_LEARNING_DIR = "08_deep_learning"
_ORDINAL_DIR = "09_ordinal_flavonol"
_ANOMALY_DIR = "10_anomaly_flavonol"
_GA_REPORTS_DIR = "12_ga_feature_selection"   # outputs/12_ga_feature_selection/<target>_<model>/ga_comparison.csv
_FLAV_COMBOS_DIR = "13_flavonol_combos"  # comparison.csv
_FINAL_COMBOS_DIR = "16_final_combos"  # comparison.csv (F1..F5 + F1n..F4n nested)

# Türkçe hedef adları (CSV daha okunaklı olsun)
_TARGET_LABELS = {
    "chlorophyll": "Klorofil",
    "flavonol": "Flavonol",
    "nbi": "NBI",
    "stress": "4 Sınıf (Sağlıklı/FD/Biyotik/Abiyotik)",
    "flavonol_pheur": "EMA ≈%3.5 Flavonoid (binary PASS/FAIL)",
}

_STAGE_LABELS = {
    _REGRESSION_DIR: "06_Regresyon",
    _REGRESSION_TUNED_DIR: "06b_Regresyon_Tuned",
    _CLASSIFICATION_DIR: "07_Stres_Siniflandirma",
    _CLASSIFICATION_RESAMPLED_DIR: "07_Stres_Siniflandirma_SMOTE",
    _CLASSIFICATION_TUNED_DIR: "07b_Stres_Siniflandirma_Tuned",
    _DEEP_LEARNING_DIR: "08_Derin_Ogrenme",
    _ORDINAL_DIR: "09_Ordinal_Flavonol",
    _ANOMALY_DIR: "10_Anomali_Tespiti",
    _GA_REPORTS_DIR: "12_GA_Feature_Selection",
    _FLAV_COMBOS_DIR: "13_Flavonol_Kombinasyonlar",
    _FINAL_COMBOS_DIR: "16_Final_Kombinasyonlar",
}

# Final tablo sütun sırası (boşlar NaN olarak kalır)
COLUMNS: list[str] = [
    "Aşama", "Hedef", "Model",
    "R²", "Accuracy", "Balanced_Acc", "Macro_F1",
    "Precision", "Recall", "F1",
    "RMSE", "RPD", "MAPE", "MAE",
    "Notlar",
]


def _model_note(stage: str, model: str, target: str) -> str:
    """Modele kısa Türkçe açıklama (rapor için)."""
    notes = {
        ("06_Regresyon", "ridge"): "L2 düzenlemesi; spektral kolinerlik altında klasik baseline",
        ("06_Regresyon", "random_forest"): "Non-lineer etkileşimler; ağaç-temelli, ölçek-bağımsız",
        ("06_Regresyon", "lightgbm"): "Gradient boosting; rezidüellere odaklanarak iteratif iyileşme",
        ("06_Regresyon", "stacking"): "Ridge+RF+LGBM → meta-Ridge, model çeşitliliğinden faydalanma",
        ("06_Regresyon", "pls"): "Spektroskopinin altın standardı; latent component bulur",
        ("06_Regresyon", "huber"): "Outlier-dirençli lineer",
        ("06_Regresyon", "elasticnet"): "L1+L2 sparse-friendly",
        ("06_Regresyon", "svr"): "RBF-kernel non-linear",
        ("06_Regresyon", "xgboost"): "Gradient boosting alternatifi",
        ("07_Stres_Siniflandirma", "random_forest"): "Ağaç-temelli baseline, dengeli",
        ("07_Stres_Siniflandirma", "lightgbm"): "Dengesiz sınıflar için class_weight + boosting",
        ("07_Stres_Siniflandirma", "stacking"): "RF + LightGBM → Logistic meta",
        ("07_Stres_Siniflandirma", "pheur"): "MLP (DL) — spektral feature → dense ağa",
        ("07_Stres_Siniflandirma", "pheur_binary"): "Flavonoid ≥3.5 PASS/FAIL — EMA raporu (≈%3.5) operasyonel eşiği",
        ("08_Derin_Ogrenme", "mlp"): "En basit DL; spektral feature → dense",
        ("08_Derin_Ogrenme", "autoencoder"): "Unsupervised pretrain; gürültüden temiz temsil",
        ("08_Derin_Ogrenme", "resnet1d"): "Skip connection ile derin spektral öğrenme",
        ("08_Derin_Ogrenme", "cnn_lstm"): "Lokal CNN + spektral sıra",
        ("08_Derin_Ogrenme", "transformer"): "Self-attention ile bant etkileşimi",
        ("08_Derin_Ogrenme", "cnn1d"): "Lokal spektral patern öğrenme",
        ("08_Derin_Ogrenme", "rnn"): "Spektral sıra → ama bantlar gerçekten zaman değil",
        ("10_Anomali_Tespiti", "autoencoder"): "Reconstruction error → gürültü-temiz temsil; Recall kritik",
        ("10_Anomali_Tespiti", "one_class_svm"): "Kernel tabanlı sınır öğrenme",
        ("10_Anomali_Tespiti", "isolation_forest"): "Ağaç tabanlı izolasyon — hızlı baseline",
    }
    return notes.get((stage, model.lower()), "")


def _read_metrics_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("metrics.json okunamadı: %s — %s", path, exc)
        return {}


def _row_from_regression(stage: str, model: str, target: str, m: dict) -> dict:
    return {
        "Aşama": stage,
        "Hedef": _TARGET_LABELS.get(target, target),
        "Model": model,
        "R²": m.get("R2"),
        "Accuracy": np.nan,
        "Balanced_Acc": np.nan,
        "Macro_F1": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "RMSE": m.get("RMSE"),
        "RPD": m.get("RPD"),
        "MAPE": m.get("MAPE"),
        "MAE": m.get("MAE"),
        "Notlar": _model_note(stage, model, target),
    }


def _row_from_classification(stage: str, model: str, target: str, m: dict) -> dict:
    return {
        "Aşama": stage,
        "Hedef": _TARGET_LABELS.get(target, target),
        "Model": model,
        "R²": np.nan,
        "Accuracy": m.get("accuracy"),
        "Balanced_Acc": m.get("balanced_accuracy"),
        "Macro_F1": m.get("macro_f1"),
        "Precision": m.get("macro_precision"),
        "Recall": m.get("macro_recall"),
        "F1": m.get("macro_f1"),
        "RMSE": np.nan,
        "RPD": np.nan,
        "MAPE": np.nan,
        "MAE": np.nan,
        "Notlar": _model_note(stage, model, target),
    }


def _scan_metrics_dir(stage_dir_name: str, task: str) -> list[dict]:
    """``outputs/<stage>/<model>/<target>/metrics.json`` yapısını tara."""
    rows: list[dict] = []
    base = paths.OUTPUTS_DIR / stage_dir_name
    if not base.exists():
        return rows
    stage_label = _STAGE_LABELS.get(stage_dir_name, stage_dir_name)
    for metrics_path in sorted(base.glob("*/*/metrics.json")):
        model = metrics_path.parent.parent.name
        target = metrics_path.parent.name
        m = _read_metrics_json(metrics_path)
        if not m:
            continue
        if task == "regression":
            rows.append(_row_from_regression(stage_label, model, target, m))
        else:
            rows.append(_row_from_classification(stage_label, model, target, m))
    return rows


def _scan_ordinal() -> list[dict]:
    """09_ordinal_flavonol/ordinal_flavonol_report.csv'den özet (mean satırları)."""
    rows: list[dict] = []
    csv_path = paths.OUTPUTS_DIR / _ORDINAL_DIR / "ordinal_flavonol_report.csv"
    if not csv_path.exists():
        return rows
    df = pd.read_csv(csv_path)
    df = df[df["fold"].astype(str) == "mean"]
    stage_label = _STAGE_LABELS[_ORDINAL_DIR]
    for _, r in df.iterrows():
        scheme = r.get("scheme", "")
        rows.append({
            "Aşama": stage_label,
            "Hedef": f"Ordinal Flavonol ({scheme})",
            "Model": str(r.get("label") or r.get("model")),
            "R²": np.nan,
            "Accuracy": float(r.get("accuracy", np.nan)),
            "Balanced_Acc": float(r.get("balanced_accuracy", np.nan)),
            "Macro_F1": float(r.get("macro_f1", np.nan)),
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": float(r.get("macro_f1", np.nan)),
            "RMSE": np.nan,
            "RPD": np.nan,
            "MAPE": np.nan,
            "MAE": float(r.get("mae", np.nan)),
            "Notlar": f"QWK={r.get('qwk', float('nan')):.3f}; {scheme} sınıf şeması",
        })
    return rows


def _scan_ga_comparisons() -> list[dict]:
    """outputs/12_ga_feature_selection/<target>_<model>/ga_comparison.csv tarayıcı.

    Her dosya, GA-seçili feature subset üzerinde 7 farklı regresörün CV
    performansını içerir. Bu, ``GA+<regressor>`` satırı olarak konsolide
    tabloya eklenir (R²_ga, RMSE_ga, RPD_ga sütunlarından).
    """
    rows: list[dict] = []
    base = paths.OUTPUTS_DIR / _GA_REPORTS_DIR
    if not base.exists():
        return rows
    stage_label = _STAGE_LABELS[_GA_REPORTS_DIR]
    for csv_path in sorted(base.glob("*/ga_comparison.csv")):
        # Klasör adı örn: "flavonol_ridge" → target=flavonol, ga_model=ridge
        parts = csv_path.parent.name.split("_", 1)
        target = parts[0] if parts else "?"
        ga_model = parts[1] if len(parts) > 1 else "?"
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            log.warning("ga_comparison okunamadı: %s — %s", csv_path, exc)
            continue
        for _, r in df.iterrows():
            rows.append({
                "Aşama": stage_label,
                "Hedef": _TARGET_LABELS.get(target, target),
                "Model": f"GA[{ga_model}] + {r.get('model')}",
                "R²": float(r.get("R2_ga", np.nan)),
                "Accuracy": np.nan,
                "Balanced_Acc": np.nan,
                "Macro_F1": np.nan,
                "Precision": np.nan,
                "Recall": np.nan,
                "F1": np.nan,
                "RMSE": float(r.get("RMSE_ga", np.nan)),
                "RPD": float(r.get("RPD_ga", np.nan)),
                "MAPE": np.nan,
                "MAE": np.nan,
                "Notlar": (
                    f"GA-seçili {int(r.get('n_features_ga', 0))}/{int(r.get('n_features_full', 0))} "
                    f"feature; ΔR²={float(r.get('delta_R2', 0)):+.3f}"
                ),
            })
    return rows


def _scan_flavonol_combos() -> list[dict]:
    """outputs/13_flavonol_combos/comparison.csv — strateji × model satırları."""
    rows: list[dict] = []
    csv_path = paths.OUTPUTS_DIR / _FLAV_COMBOS_DIR / "comparison.csv"
    if not csv_path.exists():
        return rows
    try:
        # Ayraç oto-algıla (üretici virgül kullanıyor; eski sürüm ';' idi)
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception as exc:
        log.warning("flavonol_combos comparison.csv okunamadı: %s", exc)
        return rows

    stage_label = _STAGE_LABELS[_FLAV_COMBOS_DIR]
    for _, r in df.iterrows():
        # "regression_to_ordinal :: GA+PLS → 3-sınıf" gibi sınıflandırma
        # satırları Accuracy/MacroF1 doldurur; regresyon satırları R²/RMSE.
        r2 = r.get("R2", np.nan)
        try:
            r2 = float(r2) if pd.notna(r2) and str(r2).strip() != "" else np.nan
        except Exception:
            r2 = np.nan
        acc = r.get("Accuracy", np.nan)
        try:
            acc = float(acc) if pd.notna(acc) and str(acc).strip() != "" else np.nan
        except Exception:
            acc = np.nan

        # RMSE/RPD çoğu zaman bilimsel notasyonla bozulmuş; güvenli parse
        def _f(col):
            v = r.get(col)
            try:
                return float(v) if pd.notna(v) and str(v).strip() != "" else np.nan
            except Exception:
                return np.nan

        strategy = str(r.get("strategy", "?"))
        model = str(r.get("model", "?"))
        n_ga = r.get("n_features_ga", "")
        delta_oof = r.get("delta_OOF_vs_baseline", "")
        notes_bits = [strategy]
        if pd.notna(n_ga) and str(n_ga).strip():
            try:
                notes_bits.append(f"n_ga={int(float(n_ga))}")
            except Exception:
                pass
        if pd.notna(delta_oof) and str(delta_oof).strip():
            notes_bits.append(f"ΔR²_OOF={delta_oof}")

        rows.append({
            "Aşama": stage_label,
            "Hedef": "Flavonol (GA kombinasyonları)",
            "Model": model,
            "R²": r2,
            "Accuracy": acc,
            "Balanced_Acc": _f("BalancedAcc"),
            "Macro_F1": _f("MacroF1"),
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": _f("MacroF1"),
            "RMSE": _f("RMSE"),
            "RPD": np.nan,  # comparison.csv'deki RPD sayıları bilimsel-bozuk; çekme
            "MAPE": np.nan,
            "MAE": np.nan,
            "Notlar": " | ".join(notes_bits),
        })
    return rows


def _scan_anomaly() -> list[dict]:
    """10_anomaly_flavonol/anomaly_flavonol_report.csv'den özet."""
    rows: list[dict] = []
    csv_path = paths.OUTPUTS_DIR / _ANOMALY_DIR / "anomaly_flavonol_report.csv"
    if not csv_path.exists():
        return rows
    df = pd.read_csv(csv_path)
    stage_label = _STAGE_LABELS[_ANOMALY_DIR]
    for _, r in df.iterrows():
        det = str(r.get("detector"))
        rows.append({
            "Aşama": stage_label,
            "Hedef": "Flavonoid Anomali (EMA ≈%3.5 eşiği)",
            "Model": det,
            "R²": np.nan,
            "Accuracy": np.nan,
            "Balanced_Acc": np.nan,
            "Macro_F1": np.nan,
            "Precision": float(r.get("precision", np.nan)),
            "Recall": float(r.get("recall", np.nan)),
            "F1": float(r.get("f1", np.nan)),
            "RMSE": np.nan,
            "RPD": np.nan,
            "MAPE": np.nan,
            "MAE": np.nan,
            "Notlar": _model_note(stage_label, det, ""),
        })
    return rows


def _scan_final_combos() -> list[dict]:
    """16_final_combos/comparison.csv — F1..F5 (biased) + F1n..F4n (nested-CV).

    Sütunlar: code,task,model,target,n_features,metric,mean,std,bias_vs_biased.
    Kod sonu 'n' ise nested-CV (selection-bias'sız dürüst skor). ``mean`` metrik
    tipine göre R² veya Macro_F1 kolonuna yazılır.
    """
    rows: list[dict] = []
    csv_path = paths.OUTPUTS_DIR / _FINAL_COMBOS_DIR / "comparison.csv"
    if not csv_path.exists():
        return rows
    try:
        # encoding="utf-8-sig": üretici dosyayı BOM'lu yazıyor; BOM'suz okumada
        # ilk kolon adı "﻿code" olup r.get("code") None döner → kod "?" görünür.
        df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception as exc:
        log.warning("final_combos comparison.csv okunamadı: %s", exc)
        return rows

    stage_label = _STAGE_LABELS[_FINAL_COMBOS_DIR]

    def _f(v):
        try:
            return float(v) if pd.notna(v) and str(v).strip() != "" else np.nan
        except Exception:
            return np.nan

    for _, r in df.iterrows():
        code = str(r.get("code", "?"))
        if str(r.get("status", "")).strip() in ("failed", "skipped"):
            continue
        nested = code.endswith("n")
        metric = str(r.get("metric", "")).strip()
        mean = _f(r.get("mean"))
        is_r2 = metric.upper() == "R2"
        target = str(r.get("target", "")).strip()

        # Notlar: dürüst/biased etiketi + (varsa) selection-bias büyüklüğü
        note_bits = ["nested-CV (dürüst)" if nested else "biased (tüm-veride GA)"]
        bias = _f(r.get("bias_vs_biased"))
        if not np.isnan(bias):
            note_bits.append(f"Δbias={bias:+.4f}")
        n_feat = r.get("n_features")
        if pd.notna(n_feat) and str(n_feat).strip():
            note_bits.append(f"n_feat={n_feat}")

        rows.append({
            "Aşama": stage_label,
            "Hedef": _TARGET_LABELS.get(target, target),
            "Model": f"{code}: {r.get('model', '?')}",
            "R²": mean if is_r2 else np.nan,
            "Accuracy": np.nan,
            "Balanced_Acc": np.nan,
            "Macro_F1": np.nan if is_r2 else mean,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan if is_r2 else mean,
            "RMSE": np.nan,
            "RPD": np.nan,
            "MAPE": np.nan,
            "MAE": np.nan,
            "Notlar": " | ".join(note_bits),
        })
    return rows


def _round_numeric(df: pd.DataFrame, ndigits: int = 3) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype.kind in "fc":
            df[col] = df[col].round(ndigits)
    return df


def _save_xlsx(df: pd.DataFrame, path: Path) -> bool:
    """openpyxl varsa Excel'e yaz, başarı durumunu döndür."""
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Tum_Modeller", index=False)
        log.info("Excel yazıldı: %s", path)
        return True
    except Exception as exc:
        log.warning("Excel yazılamadı (%s); CSV yeterli olacak", exc)
        return False


def run(cfg=None) -> Path:
    """Pipeline aşaması: tüm metriklerin konsolide tablosunu üret."""
    rows: list[dict] = []
    rows += _scan_metrics_dir(_REGRESSION_DIR, task="regression")
    rows += _scan_metrics_dir(_CLASSIFICATION_DIR, task="classification")
    rows += _scan_metrics_dir(_DEEP_LEARNING_DIR, task="classification")
    # GÖREV 1 — SMOTE+Tomek ile yeniden değerlendirilmiş sınıflandırma
    rows += _scan_metrics_dir(_CLASSIFICATION_RESAMPLED_DIR, task="classification")
    # GÖREV 2 — Coarse + Fine HP tuning sonuçları
    rows += _scan_metrics_dir(_REGRESSION_TUNED_DIR, task="regression")
    rows += _scan_metrics_dir(_CLASSIFICATION_TUNED_DIR, task="classification")
    rows += _scan_ordinal()
    rows += _scan_anomaly()
    rows += _scan_ga_comparisons()
    rows += _scan_flavonol_combos()
    rows += _scan_final_combos()

    if not rows:
        log.warning("Hiçbir model çıktısı bulunamadı; konsolide rapor atlanıyor")
        return paths.stage_dir("14_model_summary")

    df = pd.DataFrame(rows, columns=COLUMNS)

    # Aşama → R² azalan / Accuracy azalan şeklinde sırala
    sort_key = df["R²"].fillna(-np.inf).combine(
        df["Macro_F1"].fillna(-np.inf), max
    )
    df = (
        df.assign(_sort=sort_key)
          .sort_values(["Aşama", "_sort"], ascending=[True, False])
          .drop(columns=["_sort"])
          .reset_index(drop=True)
    )
    df = _round_numeric(df, ndigits=3)

    # Tek kaynak: numaralı aşama dizini (eski reports/model_summary kaldırıldı).
    stage_dir = paths.stage_dir("14_model_summary")
    out_dir = stage_dir

    # CSV (utf-8-sig → Excel'de Türkçe karakterler)
    csv_path = out_dir / "all_models.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Excel (opsiyonel)
    _save_xlsx(df, out_dir / "all_models.xlsx")

    # Türkçe kısa özet
    summary = [
        "Konsolide Model Karşılaştırma Tablosu",
        "=" * 60,
        f"Toplam satır       : {len(df)}",
        f"Aşamalar           : {sorted(df['Aşama'].unique().tolist())}",
        f"Hedefler           : {sorted(df['Hedef'].unique().tolist())}",
        "",
        "En iyi 5 (R² varsa, yoksa Macro_F1):",
        "-" * 60,
    ]
    df_top = df.copy()
    df_top["best_score"] = df_top["R²"].fillna(df_top["Macro_F1"])
    df_top = df_top.sort_values("best_score", ascending=False).head(5)
    for _, r in df_top.iterrows():
        score = r.get("R²")
        score_label = "R²"
        if pd.isna(score):
            score = r.get("Macro_F1")
            score_label = "F1"
        summary.append(
            f"  {r['Aşama']:<24} | {r['Model']:<20} | {r['Hedef']:<35} "
            f"| {score_label}={score:.3f}"
        )
    summary += [
        "-" * 60,
        f"\nÇıktılar           : {out_dir}",
    ]
    summary_text = "\n".join(summary) + "\n"
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8-sig")

    log.info("Konsolide tablo: %d satır → %s", len(df), csv_path)

    paths.write_source_marker(
        stage_dir,
        producer="src/m07_ensemble/model_summary.py",
        config_source=cfg.source if cfg is not None else None,
    )
    return out_dir
