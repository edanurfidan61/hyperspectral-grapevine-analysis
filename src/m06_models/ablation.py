"""GÖREV 3 — Ablation aşaması (15_ablation).

En iyi 3 model (PLS, LightGBM, RF) ile 5 deney grubu çalıştırır:

A. PREPROCESSING — Savitzky-Golay/SNV açma-kapama (dataset rebuild gerekir)
B. FEATURE       — sadece spektrum / sadece indeks / sadece 1. türev / tam
C. INDEX         — sırayla NDVI/ARI/CRI/PRI/ZTM/SIPI/FLAVI çıkarımı
D. CLASSIFICATION — resampling/class_weight varyasyonları
E. TUNING        — HP yok / sadece coarse / coarse+fine

Her deney için (model, target) bazında 5-fold CV skoru hesaplanır
(regresyon R²; sınıflandırma F1_macro). Sonuçlar:
    outputs/15_ablation/results.csv          — tüm satırlar
    outputs/15_ablation/results.xlsx         — openpyxl varsa
    outputs/15_ablation/ablation_report.md   — Türkçe özet + en iyi/en kötü

NOT: Bu aşama tek-tek model değerlendirmesi yapar; tüm pipeline aşamalarını
yeniden çalıştırmaz. Preprocessing (A) deneyleri ``--quick`` modunda atlanır
çünkü dataset rebuild birkaç dakika sürer.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from src.core import paths
from src.core.cv import load_groups, make_cv_splitter
from src.core.logging_setup import get as get_logger
from src.m05_dataset import builder as dataset_builder
from src.m06_models import hp_spaces
from src.m06_models.registry import (
    MODELS_CLASSIFICATION,
    MODELS_REGRESSION,
    _build_model,
    _ensure_models_imported,
)
from src.m06_models.utils import SafeResampler

log = get_logger("m06_models.ablation")

# Ablation için sabitler
ABLATION_MODELS: tuple[str, ...] = ("pls", "lightgbm", "random_forest")
REG_TARGET = ("flavonol", "y_flav")              # zor hedef — varyasyonlar belirgin
CLS_TARGET = ("stress", "y_stress")              # sınıf dengesizliği var
INDEX_TO_DROP = ("NDVI", "ARI", "CRI", "PRI", "ZTM", "SIPI", "FLAVI")


# ---------------------------------------------------------------------------
# Deney tanımı (dataclass)
# ---------------------------------------------------------------------------
@dataclass
class Experiment:
    """Tek bir ablation deneyi: grup, kod, açıklama + uygulayıcı kapanış."""
    group: str
    code: str
    name: str
    apply: Callable[[dict, np.ndarray, list[str], dict], dict]
    """``apply(state, X, feature_names, ctx) -> {"X": new_X, "cfg": new_cfg, ...}``."""
    tasks: tuple[str, ...] = ("regression", "classification")


# ---------------------------------------------------------------------------
# Tek bir (deney × model × target) için CV skoru
# ---------------------------------------------------------------------------
def _score(
    task: str,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    cfg,
    *,
    groups: np.ndarray | None = None,
    override_hp: dict | None = None,
) -> tuple[float, float] | None:
    """Pipeline (Scaler [+ SafeResampler] + estimator) ile k-fold mean,std skoru."""
    cls_map = MODELS_REGRESSION if task == "regression" else MODELS_CLASSIFICATION
    if model_name not in cls_map:
        return None
    model = _build_model(cls_map[model_name], cfg)
    valid = ~np.isnan(y) if task == "regression" else np.ones(len(y), dtype=bool)
    Xv, yv = X[valid], y[valid]
    gv = groups[valid] if groups is not None else None

    est = hp_spaces.make_estimator(task=task, model_name=model_name,
                                   random_state=model.random_state)
    if est is None:
        return None
    if override_hp:
        try:
            est.set_params(**override_hp)
        except Exception:
            pass  # ilgisiz param atlandı

    # F4 NaN-bug fix: dar alt-kümede (1.türev ∩ GA) bir fold'un train kısmında
    # sıfır-varyanslı kolon kalabiliyor → StandardScaler 0'a bölünce NaN →
    # PLS "A has a NaN entry". VarianceThreshold ile fold-içi sabit kolonları at.
    steps: list = [("vt", VarianceThreshold(threshold=0.0))]
    if model.requires_scaling:
        steps.append(("scaler", StandardScaler()))
    use_resample = (
        task == "classification"
        and getattr(model, "resampling_enabled", False)
        and getattr(model, "resampling_method", "none") != "none"
    )
    if use_resample:
        from imblearn.pipeline import Pipeline as ImbPipeline
        steps.append(("resampler", SafeResampler(
            method=model.resampling_method, random_state=model.random_state)))
        steps.append(("est", est))
        pipe = ImbPipeline(steps)
    else:
        steps.append(("est", est))
        pipe = SkPipeline(steps)

    cv = make_cv_splitter(
        n_splits=model.cv, task=task, groups=gv,
        random_state=model.random_state,
        stratify_regression=model.stratify_regression,
        n_bins=model.regression_n_bins,
    )
    scoring = "r2" if task == "regression" else "f1_macro"
    try:
        if gv is not None:
            sc = cross_val_score(pipe, Xv, yv, cv=cv, scoring=scoring,
                                 n_jobs=-1, groups=gv, error_score="raise")
        else:
            sc = cross_val_score(pipe, Xv, yv, cv=cv, scoring=scoring,
                                 n_jobs=-1, error_score="raise")
        return float(np.mean(sc)), float(np.std(sc))
    except Exception as exc:
        log.warning("%s/%s: CV skoru hesaplanamadı — %s", model_name, task, exc)
        return None


# ---------------------------------------------------------------------------
# Yardımcılar — feature mask (B, C deneyleri için)
# ---------------------------------------------------------------------------
def _is_spectrum(name: str) -> bool:
    n = name.lower()
    return n.startswith(("snv_r", "r_", "spec_")) or "spec" in n


def _is_deriv(name: str) -> bool:
    return name.lower().startswith(("d1snv_r", "d1_", "deriv"))


def _is_index(name: str) -> bool:
    return not (_is_spectrum(name) or _is_deriv(name))


def _mask_by_kind(feature_names: list[str], kind: str) -> np.ndarray:
    """Feature mask: "spectrum"|"indices"|"deriv"|"full"."""
    if kind == "full":
        return np.ones(len(feature_names), dtype=bool)
    out = np.zeros(len(feature_names), dtype=bool)
    for i, n in enumerate(feature_names):
        if kind == "spectrum" and _is_spectrum(n):
            out[i] = True
        elif kind == "indices" and _is_index(n):
            out[i] = True
        elif kind == "deriv" and _is_deriv(n):
            out[i] = True
    return out


def _mask_drop_index(feature_names: list[str], idx_name: str) -> np.ndarray:
    """Belirli bir indeks adıyla başlayan tüm kolonları çıkar (ör. 'NDVI')."""
    out = np.ones(len(feature_names), dtype=bool)
    needle = idx_name.lower()
    for i, n in enumerate(feature_names):
        ln = n.lower()
        # "ndvi" ya da "ndvi_mean" gibi kolonları çıkar; "snv_r..." gibi spektral
        # kolonları yanlışlıkla çıkarmamak için index-benzeri olduklarından emin ol.
        if ln.startswith(needle) and _is_index(n):
            out[i] = False
    return out


# ---------------------------------------------------------------------------
# Deney apply'ları
# ---------------------------------------------------------------------------
def _apply_full(state, X, feature_names, ctx):
    return {"X": X, "cfg": ctx["cfg"]}


def _apply_feature_subset(kind: str):
    def _go(state, X, feature_names, ctx):
        mask = _mask_by_kind(feature_names, kind)
        return {"X": X[:, mask], "cfg": ctx["cfg"],
                "n_features": int(mask.sum())}
    return _go


def _apply_drop_index(idx_name: str):
    def _go(state, X, feature_names, ctx):
        mask = _mask_drop_index(feature_names, idx_name)
        return {"X": X[:, mask], "cfg": ctx["cfg"],
                "n_features": int(mask.sum())}
    return _go


def _apply_cls_resampling(method: str | None, class_weight: str | None):
    """method=None → resampling kapalı; class_weight=None → balanced kapalı."""
    def _go(state, X, feature_names, ctx):
        cfg2 = _cfg_clone(ctx["cfg"])
        cfg2.set("classification.resampling.enabled", method is not None)
        cfg2.set("classification.resampling.method", method or "none")
        # override_hp: RF/LGBM için class_weight=None
        override_hp = {"class_weight": class_weight} if class_weight != "_keep" else None
        return {"X": X, "cfg": cfg2, "override_hp": override_hp}
    return _go


def _apply_preprocessing(savgol_off: bool, snv_off: bool):
    """Dataset rebuild ister; çağıran yapı bunu görür ve rebuild eder."""
    def _go(state, X, feature_names, ctx):
        cfg2 = _cfg_clone(ctx["cfg"])
        if savgol_off:
            # window_length=1 + polyorder=0 ≈ filtre devre dışı
            cfg2.set("preprocessing.savgol.window_length", 1)
            cfg2.set("preprocessing.savgol.polyorder", 0)
        if snv_off:
            cfg2.set("preprocessing.snv.enabled", False)
        # Caller, rebuild_needed=True görünce dataset_builder.build(cfg2, force=True) yapar.
        return {"rebuild_needed": True, "cfg": cfg2}
    return _go


def _apply_tuning(level: str):
    """level: "none" | "coarse" | "fine" → ctx['tuning_level'] olarak işaretle."""
    def _go(state, X, feature_names, ctx):
        return {"X": X, "cfg": ctx["cfg"], "tuning_level": level}
    return _go


def _cfg_clone(cfg):
    """Mevcut Config'in yüzeysel kopyasını döndür (set() override için)."""
    from src.core.config import Config
    return Config(copy.deepcopy(cfg.as_dict()), source=cfg.source)


# ---------------------------------------------------------------------------
# Deney listesi (plandaki tüm matris)
# ---------------------------------------------------------------------------
def _build_experiments() -> list[Experiment]:
    exps: list[Experiment] = []

    # Baseline (tam)
    exps.append(Experiment("BASE", "B0", "Baseline (tam)", _apply_full))

    # A. PREPROCESSING (dataset rebuild gerektirir)
    exps += [
        Experiment("PREPROC", "A1", "SavGol kapalı", _apply_preprocessing(True, False)),
        Experiment("PREPROC", "A2", "SNV kapalı", _apply_preprocessing(False, True)),
        Experiment("PREPROC", "A3", "SavGol+SNV kapalı", _apply_preprocessing(True, True)),
    ]

    # B. FEATURE
    exps += [
        Experiment("FEATURE", "B1", "Sadece spektrum", _apply_feature_subset("spectrum")),
        Experiment("FEATURE", "B2", "Sadece indeks", _apply_feature_subset("indices")),
        Experiment("FEATURE", "B4", "Sadece 1. türev", _apply_feature_subset("deriv")),
    ]
    # B3 ≡ Baseline; tekrar yok

    # C. INDEX (her birini sırayla çıkar)
    for i, idx in enumerate(INDEX_TO_DROP, start=1):
        exps.append(Experiment("INDEX", f"C{i}", f"{idx} çıkarıldı", _apply_drop_index(idx)))

    # D. CLASSIFICATION (sadece classification task)
    exps += [
        Experiment("CLS", "D1", "Resampling YOK", _apply_cls_resampling(None, "_keep"),
                   tasks=("classification",)),
        Experiment("CLS", "D2", "Sadece SMOTE", _apply_cls_resampling("smote", "_keep"),
                   tasks=("classification",)),
        Experiment("CLS", "D3", "Class weight YOK", _apply_cls_resampling("smote_tomek", None),
                   tasks=("classification",)),
        Experiment("CLS", "D4", "Tam baseline (smote_tomek + balanced)",
                   _apply_cls_resampling("smote_tomek", "_keep"),
                   tasks=("classification",)),
    ]

    # E. TUNING
    exps += [
        Experiment("TUNING", "E1", "HP yok (sklearn default)", _apply_tuning("none")),
        Experiment("TUNING", "E2", "Sadece coarse", _apply_tuning("coarse")),
        Experiment("TUNING", "E3", "Coarse + fine", _apply_tuning("fine")),
    ]

    return exps


# ---------------------------------------------------------------------------
# Ana koşucu
# ---------------------------------------------------------------------------
def _resolve_dataset(cfg, rebuild_needed: bool, exp_cfg):
    """Gerekirse dataset rebuild yap; her halükarda X/feature_names döndür."""
    if rebuild_needed:
        log.info("Preprocessing ablation: dataset_builder rebuild ediliyor...")
        dataset_builder.build(exp_cfg, force=True)
    data = dataset_builder.load()
    return data["X"], data["feature_names"], data


def _tuning_score(
    task: str, model_name: str, X: np.ndarray, y: np.ndarray, cfg,
    *, level: str, groups: np.ndarray | None,
    n_iter: int, n_trials: int, timeout: int,
) -> tuple[float, float] | None:
    """E grubu için: HP seviyesi (none/coarse/fine) ile tek skor."""
    if level == "none":
        return _score(task, model_name, X, y, cfg, groups=groups)

    # Aşama A — coarse (her seviyede çalışır)
    cls_map = MODELS_REGRESSION if task == "regression" else MODELS_CLASSIFICATION
    if model_name not in cls_map:
        return None
    model = _build_model(cls_map[model_name], cfg)
    param_dist = hp_spaces.get_param_dist(task, model_name)
    if param_dist is None:
        return _score(task, model_name, X, y, cfg, groups=groups)
    base_est = hp_spaces.make_estimator(task, model_name, model.random_state)

    valid = ~np.isnan(y) if task == "regression" else np.ones(len(y), dtype=bool)
    Xv, yv = X[valid], y[valid]
    gv = groups[valid] if groups is not None else None
    scoring = "r2" if task == "regression" else "f1_macro"

    try:
        best_c, _score_c = model.tune_coarse(
            Xv, yv, groups=gv, param_dist=param_dist,
            n_iter=n_iter, scoring=scoring, estimator=base_est,
        )
    except Exception as exc:
        log.warning("Coarse başarısız %s/%s: %s", task, model_name, exc)
        return None

    # GÖREV 7: tuning sonrası HP ile AYNI CV protokolü → tutarlı (mean, std)
    # RandomizedSearchCV.best_score_ skaler mean döndürüyor (std=0 sahte);
    # _score zaten cross_val_score.mean()+std() döndürür.
    if level == "coarse":
        return _score(task, model_name, X, y, cfg,
                      groups=groups, override_hp=best_c)

    # level == "fine" → Optuna ile dar alan
    from src.m06_models import optuna_tuner

    def _factory(params):
        est = hp_spaces.make_estimator(task, model_name, model.random_state)
        return est.set_params(**params)

    try:
        best_f, _score_f = optuna_tuner.fine_tune(
            model, Xv, yv, coarse_best=best_c, param_dist=param_dist,
            n_trials=n_trials, timeout=timeout, scoring=scoring,
            groups=gv, estimator_factory=_factory,
        )
        return _score(task, model_name, X, y, cfg,
                      groups=groups, override_hp=best_f or best_c)
    except Exception as exc:
        log.warning("Fine başarısız %s/%s: %s", task, model_name, exc)
        return _score(task, model_name, X, y, cfg,
                      groups=groups, override_hp=best_c)


def run(cfg) -> Path:
    """15_ablation aşaması — tüm deneyleri yürüt ve rapor üret."""
    t0 = time.time()
    _ensure_models_imported()
    out_dir = paths.stage_dir("15_ablation")
    paths.write_source_marker(
        out_dir, producer="src/m06_models/ablation.py",
        config_source=cfg.source,
    )

    quick = bool(cfg.get("quick", False))
    n_iter = int(cfg.get("hp_search.coarse.n_iter", 40))
    n_trials = int(cfg.get("hp_search.fine.n_trials", 50))
    timeout = int(cfg.get("hp_search.fine.timeout_seconds", 600))
    if quick:
        n_iter, n_trials, timeout = min(6, n_iter), min(8, n_trials), min(60, timeout)

    # Ortak: baseline dataset + groups
    base_data = dataset_builder.load()
    feature_names = base_data["feature_names"]
    group_key = str(cfg.get("cv.group_key", "leaf"))
    groups_base = load_groups(paths.OUTPUTS_DIR / "01_dataset", key=group_key)

    experiments = _build_experiments()
    if quick:
        # Her grupta ilk deneyi + baseline'ı bırak
        keep = {"BASE"}
        seen = set()
        slim = []
        for e in experiments:
            if e.group in keep:
                slim.append(e); continue
            if e.group not in seen:
                slim.append(e); seen.add(e.group)
        experiments = slim
        log.info("QUICK modu: %d deney (her gruptan 1)", len(experiments))

    rows: list[dict] = []
    skip_rebuild = quick  # quick'te PREPROCESSING'i atla
    # PREPROC deneyleri (A1/A3 savgol-kapalı) canonical 01_dataset'i force=True
    # ile EZER → türevler sıfırlanır. Bayrak: döngü sonunda orijinal cfg ile
    # geri kurmak için. Aksi halde sonraki aşamalar/çalıştırmalar ölü türev görür.
    dataset_clobbered = False
    for exp in experiments:
        log.info("[%s/%s] %s", exp.group, exp.code, exp.name)
        ctx = {"cfg": cfg}
        out = exp.apply({}, base_data["X"], feature_names, ctx)
        # Dataset rebuild lazım mı?
        if out.get("rebuild_needed"):
            if skip_rebuild:
                log.info("  → quick modu: rebuild atlandı")
                continue
            dataset_clobbered = True
            X_exp, fn_exp, data_exp = _resolve_dataset(cfg, True, out["cfg"])
            groups_exp = load_groups(paths.OUTPUTS_DIR / "01_dataset", key=group_key)
        else:
            X_exp = out.get("X", base_data["X"])
            fn_exp = feature_names
            data_exp = base_data
            groups_exp = groups_base
        exp_cfg = out.get("cfg", cfg)
        override_hp = out.get("override_hp")
        tuning_level = out.get("tuning_level")  # E grubu

        for model_name in ABLATION_MODELS:
            for task, tgt in (("regression", REG_TARGET), ("classification", CLS_TARGET)):
                if task not in exp.tasks:
                    continue
                y = data_exp[tgt[1]]
                if tuning_level is not None:
                    res = _tuning_score(task, model_name, X_exp, y, exp_cfg,
                                        level=tuning_level, groups=groups_exp,
                                        n_iter=n_iter, n_trials=n_trials, timeout=timeout)
                else:
                    res = _score(task, model_name, X_exp, y, exp_cfg,
                                 groups=groups_exp, override_hp=override_hp)
                if res is None:
                    continue
                mean, std = res
                rows.append({
                    "group": exp.group, "code": exp.code, "experiment": exp.name,
                    "model": model_name, "task": task, "target": tgt[0],
                    "metric": "R2" if task == "regression" else "Macro_F1",
                    "mean": round(mean, 4), "std": round(std, 4),
                    "n_features": out.get("n_features", X_exp.shape[1]),
                })

    # PREPROC deneyleri 01_dataset'i bozduysa orijinal cfg ile geri kur — böylece
    # sonraki aşamalar (12_ga, 16_final_combos) ve sonraki çalıştırmalar doğru
    # türevli veriyi görür. (Bu olmadan F4/türev sonuçları sessizce çürür.)
    if dataset_clobbered:
        log.info("PREPROC ablation sonrası 01_dataset orijinal cfg ile geri kuruluyor...")
        try:
            dataset_builder.build(cfg, force=True)
        except Exception as exc:
            log.error("01_dataset geri kurulamadı! Manuel rebuild gerekli: %s", exc)

    if not rows:
        log.warning("Ablation: hiç sonuç yok")
        return out_dir

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False, encoding="utf-8-sig")
    try:
        df.to_excel(out_dir / "results.xlsx", index=False)
    except Exception as exc:
        log.warning("xlsx yazılamadı (%s); CSV yeterli", exc)

    # Türkçe rapor
    md = ["# Ablation Raporu (15_ablation)", "",
          f"Toplam deney satırı: **{len(df)}**",
          f"Süre: **{time.time() - t0:.1f}s**", ""]
    # Her grup için: en iyi 3 / en kötü 3
    for grp, gdf in df.groupby("group"):
        md.append(f"## {grp}")
        for task, tdf in gdf.groupby("task"):
            metric = tdf["metric"].iloc[0]
            top = tdf.nlargest(3, "mean")[["code", "model", "experiment", "mean", "std"]].copy()
            # GÖREV 7: mean ± std okunur biçimde göster
            top["score"] = top.apply(
                lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1,
            )
            top = top[["code", "model", "experiment", "score"]]
            md.append(f"### {task} (en iyi 3, {metric})")
            md.append(top.to_markdown(index=False))
            md.append("")
    (out_dir / "ablation_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("Ablation tamam: %d satır → %s", len(df), out_dir)
    return out_dir
