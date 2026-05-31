"""Flavonol regresyonu için kombinasyon stratejileri (v2).

v1 sonuçları gösterdi ki:
  - GA+PLS gerçek performansı ölçüm yöntemine bağlı:
    * cross_val_score (fold-mean R²)        → ~0.556 (GA modülünün yöntemi)
    * cross_val_predict (OOF birleşik R²)   → ~0.450
    Aynı veri, aynı CV, farklı sayı! Bu modül her ikisini birden raporlar.

  - Hazır GA maskesi ham bantlar üzerinde optimize edilmişti. Ön işleme
    spektral yapıyı değiştirdiği için "preproc + hazır GA" ÇÖKTÜ. Doğrusu:
    HER ön işleme için GA'yı sıfırdan çalıştırmak.

  - Tek seed (42) GA stochastik. 3 farklı seed → maskelerin union/
    intersection kümesi → daha sağlam alt küme.

v2'deki stratejiler:
    1) ``residual_pls_lgbm``   — GA+PLS lineer + LGBM rezidüel
    2) ``stacking_ga``         — GA+{PLS, SVR, LGBM, Ridge} OOF + Ridge meta
    3) ``boosting_ga``         — GA + XGBoost / CatBoost
    4) ``preproc_ga_pls_v2``   — Her ön işleme için GA SIFIRDAN
    5) ``multiseed_ga_pls``    — 3 seed × GA → seed/union/intersection
    6) ``regression_to_ordinal`` — Dengeli (kvantil) sınırlarla 3 sınıf

Tüm sonuçlar iki R² değeriyle raporlanır:
    R2_fold_mean = fold-bazlı R² ortalaması (GA modülüyle uyumlu)
    R2          = OOF birleşik R² (konservatif)

CLI:
    python -m src.m06_models.flavonol_combos
    python -m src.m06_models.flavonol_combos --strategies multiseed_ga_pls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.core import paths
from src.core.logging_setup import get as get_logger
from src.m02_preprocessing import spectral as sp

log = get_logger("m06_models.flavonol_combos")

SEED = 42
N_SPLITS = 5

# Hazır GA maskesi (12_ga_feature_selection, pop=150 ngen=100, R²_fold_mean=0.556)
GA_MASK_PATH = paths.OUTPUTS_DIR / "12_ga_feature_selection" / "flavonol_pls" / "ga_best_mask.npy"

STAGE_DIR = paths.OUTPUTS_DIR / "13_flavonol_combos"
CACHE_DIR = STAGE_DIR / "cache"

# v2'de yeni GA çalıştırmaları için (hız için orijinalden küçük) parametreler.
# pop=100, ngen=60 → ~60 sn × n koşu. 6 preproc + 2 yeni seed = 8 koşu ≈ 8 dk.
GA_POP_V2 = 100
GA_NGEN_V2 = 60


# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------
def _load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X, y_flavonol ve hazır GA maskesini yükle (NaN'leri filtrele)."""
    ds = paths.OUTPUTS_DIR / "01_dataset"
    x_path, y_path = ds / "X.npy", ds / "y_flav.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Dataset eksik: {x_path}, {y_path}. "
            f"Önce: python main.py --stages 01_dataset"
        )
    if not GA_MASK_PATH.exists():
        raise FileNotFoundError(
            f"GA maskesi yok: {GA_MASK_PATH}. "
            f"Önce: python -m src.m04_features.ga_feature_selection --target flavonol --model pls"
        )

    X = np.load(x_path)
    y = np.load(y_path)
    mask = np.load(GA_MASK_PATH).astype(bool)

    valid = np.isfinite(y)
    X, y = X[valid], y[valid]

    if mask.shape[0] != X.shape[1]:
        raise ValueError(
            f"GA maskesi ({mask.shape[0]}) ile X bant sayısı ({X.shape[1]}) eşleşmiyor."
        )
    log.info("Yüklendi: X=%s, y=%s, hazır GA bant=%d", X.shape, y.shape, int(mask.sum()))
    return X, y, mask


# ---------------------------------------------------------------------------
# Metrikler — iki yöntemli rapor
# ---------------------------------------------------------------------------
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """OOF birleşik metrikler: R², RMSE, RPD, MAPE."""
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rpd = float(np.std(y_true) / rmse) if rmse > 0 else 0.0
    nz = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100)
        if int(nz.sum()) > 0 else 0.0
    )
    return {"R2": r2, "RMSE": rmse, "RPD": rpd, "MAPE": mape}


def _make_cv() -> KFold:
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def _eval_dual(model_factory: Callable[[], object],
               X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Hem fold-mean R² (GA-uyumlu) hem OOF metriklerini hesapla."""
    cv = _make_cv()
    fold_r2 = float(cross_val_score(
        model_factory(), X, y, cv=cv, scoring="r2", n_jobs=1
    ).mean())
    y_pred = cross_val_predict(model_factory(), X, y, cv=cv, n_jobs=1)
    m = _metrics(y, y_pred)
    m["R2_fold_mean"] = fold_r2
    return m


# ---------------------------------------------------------------------------
# Regresörler
# ---------------------------------------------------------------------------
def _pls(n_components: int) -> PLSRegression:
    """GA modülüyle birebir uyumlu PLS: built-in scaling, scale=True.

    NOT: GA modülü ``PLSRegression(n_components=...)`` kullanıyor, scale
    parametresine dokunmuyor → sklearn varsayılanı scale=True. Aynısını
    burada da kullanıyoruz, böylece baseline R² rakamı GA özet dosyasıyla
    örtüşür (~0.556 civarı).
    """
    return PLSRegression(n_components=n_components, scale=True)


def _scaled_pls(n_components: int) -> PLSRegression:
    """PLS'in kendi scaling'i var; harici StandardScaler GEREKMİYOR.

    İsim 'scaled' v1'den miras (geriye uyumluluk için adı koruyoruz).
    """
    return _pls(n_components)


def _scaled_svr() -> SkPipeline:
    return SkPipeline([("scaler", StandardScaler()),
                       ("model", SVR(kernel="rbf", C=1.0, gamma="scale"))])


def _lgbm(**kw):
    from lightgbm import LGBMRegressor
    base = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                min_data_in_leaf=5, random_state=SEED, n_jobs=1,
                verbose=-1, verbosity=-1, force_col_wise=True)
    base.update(kw)
    return LGBMRegressor(**base)


def _scaled_ridge(alpha: float = 1.0) -> SkPipeline:
    return SkPipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))])


def _pls_n_components(n_features: int, n_samples: int, target: int = 10) -> int:
    """GA modülüyle uyumlu üst sınır (target=10)."""
    return max(1, min(target, n_features, n_samples - 1))


# ---------------------------------------------------------------------------
# GA cache yardımcısı — yeniden çalıştırmadan önce diskten oku
# ---------------------------------------------------------------------------
def _safe_key(s: str) -> str:
    out = s.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out = out.replace("+", "_").replace("/", "_").replace("-", "_")
    out = "".join(ch for ch in out if ch.isalnum() or ch == "_")
    return out


def _run_ga_cached(X_used: np.ndarray, y: np.ndarray,
                   cache_key: str, seed: int = SEED,
                   pop: int = GA_POP_V2, ngen: int = GA_NGEN_V2) -> np.ndarray:
    """GA'yı çalıştır veya cache'den yükle.

    Cache dosyası: ``outputs/13_flavonol_combos/cache/ga_<key>_seed<s>.npy``
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"ga_{cache_key}_seed{seed}_pop{pop}_ngen{ngen}.npy"
    if cache_path.exists():
        log.info("  GA cache HIT: %s", cache_path.name)
        return np.load(cache_path).astype(bool)

    log.info("  GA çalışıyor: key=%s seed=%d pop=%d ngen=%d", cache_key, seed, pop, ngen)
    from src.m04_features import ga_feature_selection as ga_mod
    t0 = time.time()
    mask, _, _ = ga_mod._run_ga(
        X_used, y, model="pls", pop_size=pop, n_gen=ngen,
        seed=seed, n_jobs=1,
    )
    log.info("  GA bitti: %.1f sn, %d bant seçildi", time.time() - t0, int(mask.sum()))
    np.save(cache_path, mask)
    return mask


# ---------------------------------------------------------------------------
# Strateji 1 — Residual: PLS + LGBM
# ---------------------------------------------------------------------------
def strategy_residual_pls_lgbm(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    """İki aşamalı: ŷ = PLS(x) + LGBM(x; residual)."""
    Xs = X[:, mask]
    n_comp = _pls_n_components(Xs.shape[1], len(y))
    cv = _make_cv()

    fold_r2_pls, fold_r2_combo = [], []
    y_pred_pls = np.zeros(len(y))
    y_pred_combo = np.zeros(len(y))

    for tr, te in cv.split(Xs):
        # PLS kendi scaling'ini yapıyor (scale=True); LGBM ölçek-bağımsız.
        pls = _pls(n_comp).fit(Xs[tr], y[tr])
        pls_tr = np.asarray(pls.predict(Xs[tr])).ravel()
        pls_te = np.asarray(pls.predict(Xs[te])).ravel()
        residual_tr = y[tr] - pls_tr
        lgbm = _lgbm().fit(Xs[tr], residual_tr)
        lgbm_te = lgbm.predict(Xs[te])

        y_pred_pls[te] = pls_te
        y_pred_combo[te] = pls_te + lgbm_te
        fold_r2_pls.append(r2_score(y[te], pls_te))
        fold_r2_combo.append(r2_score(y[te], pls_te + lgbm_te))

    return {
        "GA+PLS (kontrol)": {
            **_metrics(y, y_pred_pls),
            "R2_fold_mean": float(np.mean(fold_r2_pls)),
        },
        "GA+PLS + LGBM(residual)": {
            **_metrics(y, y_pred_combo),
            "R2_fold_mean": float(np.mean(fold_r2_combo)),
        },
    }


# ---------------------------------------------------------------------------
# Strateji 2 — Stacking
# ---------------------------------------------------------------------------
def strategy_stacking_ga(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    """Base modellerin OOF tahminleri → meta-Ridge."""
    Xs = X[:, mask]
    n_comp = _pls_n_components(Xs.shape[1], len(y))

    base_models: dict[str, Callable[[], object]] = {
        "pls": lambda: _scaled_pls(n_comp),
        "svr": lambda: _scaled_svr(),
        "lgbm": lambda: _lgbm(),
        "ridge": lambda: _scaled_ridge(1.0),
    }

    cv = _make_cv()
    oof = np.zeros((len(y), len(base_models)))
    out: dict[str, dict] = {}
    for j, (name, factory) in enumerate(base_models.items()):
        fold_r2 = float(cross_val_score(factory(), Xs, y, cv=cv, scoring="r2", n_jobs=1).mean())
        oof[:, j] = cross_val_predict(factory(), Xs, y, cv=cv, n_jobs=1)
        m = _metrics(y, oof[:, j])
        m["R2_fold_mean"] = fold_r2
        out[f"GA+{name} (base)"] = m

    # Meta-Ridge — OOF üstünde 5-fold
    meta_factory = lambda: Ridge(alpha=1.0)
    fold_r2_meta = float(cross_val_score(meta_factory(), oof, y, cv=cv, scoring="r2", n_jobs=1).mean())
    meta_pred = cross_val_predict(meta_factory(), oof, y, cv=cv, n_jobs=1)
    m = _metrics(y, meta_pred)
    m["R2_fold_mean"] = fold_r2_meta
    out["Stacking (GA, meta=Ridge)"] = m

    return out


# ---------------------------------------------------------------------------
# Strateji 3 — XGBoost / CatBoost
# ---------------------------------------------------------------------------
def _xgboost_model():
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None
    return XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, n_jobs=1, verbosity=0, tree_method="hist",
    )


def _catboost_model():
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        return None
    return CatBoostRegressor(
        iterations=400, learning_rate=0.05, depth=4,
        l2_leaf_reg=3.0, random_seed=SEED,
        verbose=False, allow_writing_files=False,
    )


def strategy_boosting_ga(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    Xs = X[:, mask]
    out: dict[str, dict] = {}

    if _xgboost_model() is None:
        log.warning("XGBoost yok — atlanıyor.")
    else:
        out["GA+XGBoost"] = _eval_dual(_xgboost_model, Xs, y)

    if _catboost_model() is None:
        log.warning("CatBoost yok — atlanıyor.")
    else:
        out["GA+CatBoost"] = _eval_dual(_catboost_model, Xs, y)

    if not out:
        out["(boosting kütüphanesi yok)"] = {"R2": float("nan"), "R2_fold_mean": float("nan")}
    return out


# ---------------------------------------------------------------------------
# Strateji 4 (v2) — Ön işleme öncesi GA SIFIRDAN
# ---------------------------------------------------------------------------
PREPROC_VARIANTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "raw":             lambda X: X.astype(np.float64, copy=True),
    "SG-d1":           lambda X: sp.sg_first_derivative(X, window_length=11, polyorder=2),
    "SG-d2":           lambda X: sp.savitzky_golay(X, window_length=11, polyorder=3, deriv=2),
    "SNV":             lambda X: sp.snv(X),
    "MSC":             lambda X: sp.msc(X),
    "SNV+SG-d1":       lambda X: sp.sg_first_derivative(sp.snv(X), window_length=11, polyorder=2),
}


def strategy_preproc_ga_pls_v2(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    """v1'den farkı: HER ön işleme için GA'yı sıfırdan çalıştır.

    v1 hatalıydı çünkü ham bantlar üzerinde bulunmuş GA maskesini, ön işleme
    sonrasında değişmiş spektruma uyguluyordu — bantlar artık aynı
    bilgi taşımıyor. Burada her varyant kendi GA'sını alır.
    """
    out: dict[str, dict] = {}
    for name, fn in PREPROC_VARIANTS.items():
        try:
            X_pp = fn(X)
            if X_pp.shape[1] != X.shape[1]:
                log.warning("%s bant sayısını değiştirdi, atlanıyor", name)
                continue
            cache_key = _safe_key(name)
            pp_mask = _run_ga_cached(X_pp, y, cache_key, seed=SEED)
            X_sel = X_pp[:, pp_mask]
            n_comp = _pls_n_components(int(pp_mask.sum()), len(y))
            metrics = _eval_dual(lambda nc=n_comp: _scaled_pls(nc), X_sel, y)
            metrics["n_features_ga"] = int(pp_mask.sum())
            out[f"{name} → freshGA → PLS"] = metrics
        except Exception as exc:
            log.exception("%s hatası: %s", name, exc)
            out[f"{name} → freshGA → PLS"] = {
                "R2": float("nan"), "R2_fold_mean": float("nan"),
            }
    return out


# ---------------------------------------------------------------------------
# Strateji 5 — Multi-seed GA + union/intersection
# ---------------------------------------------------------------------------
def strategy_multiseed_ga_pls(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    """3 seed × GA → seed/union/intersection maskeleriyle PLS.

    Mantık:
      - GA stochastik; tek seed maskesi gürültüye karşı kırılgan.
      - 3 seed çalıştır → seçilen bantların oy birliği → daha sağlam.
      - intersection: konservatif (kesin önemli bantlar)
      - union: liberal (potansiyel önemli her şey)
    """
    seeds = (42, 7, 123)
    masks: list[np.ndarray] = []
    for s in seeds:
        m = _run_ga_cached(X, y, "raw_multiseed", seed=s)
        masks.append(m)
        log.info("  seed=%d → %d bant", s, int(m.sum()))

    union_mask = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        union_mask |= m
    intersection_mask = np.ones_like(masks[0], dtype=bool)
    for m in masks:
        intersection_mask &= m

    out: dict[str, dict] = {}
    candidates = [
        ("seed42", masks[0]),
        ("seed7", masks[1]),
        ("seed123", masks[2]),
        ("intersection", intersection_mask),
        ("union", union_mask),
    ]
    for name, m in candidates:
        n_sel = int(m.sum())
        if n_sel < 5:
            log.warning("  %s yetersiz bant (%d), atlanıyor", name, n_sel)
            out[f"GA-{name} + PLS"] = {"R2": float("nan"), "R2_fold_mean": float("nan"),
                                        "n_features_ga": n_sel}
            continue
        n_comp = _pls_n_components(n_sel, len(y))
        metrics = _eval_dual(lambda nc=n_comp, mm=m: _scaled_pls(nc), X[:, m], y)
        metrics["n_features_ga"] = n_sel
        out[f"GA-{name} + PLS"] = metrics
    return out


# ---------------------------------------------------------------------------
# Strateji 6 — Regresyon → Ordinal (kvantil sınırlarla)
# ---------------------------------------------------------------------------
def strategy_regression_to_ordinal(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    """v1'de sabit sınırlar (1.5, 2.5) sınıfları çok dengesiz yapıyordu
    (Acc=0.93 ama BalAcc=0.42 — "her şeye düşük de" yeterli geliyordu).
    Burada DENGE için kvantil-bazlı sınırlar (33%, 67%) kullanılır.
    """
    Xs = X[:, mask]
    n_comp = _pls_n_components(Xs.shape[1], len(y))
    cv = _make_cv()
    metrics_reg = _eval_dual(lambda: _scaled_pls(n_comp), Xs, y)
    y_pred = cross_val_predict(_scaled_pls(n_comp), Xs, y, cv=cv, n_jobs=1)

    edges = tuple(float(v) for v in np.quantile(y, [1.0/3.0, 2.0/3.0]))

    def _bin(arr: np.ndarray) -> np.ndarray:
        out = np.zeros(len(arr), dtype=int)
        out[arr >= edges[0]] = 1
        out[arr >= edges[1]] = 2
        return out

    y_true_bin = _bin(y)
    y_pred_bin = _bin(y_pred)

    return {
        "GA+PLS (regresyon)": metrics_reg,
        "GA+PLS → 3-sınıf (kvantil binning)": {
            "Accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
            "BalancedAcc": float(balanced_accuracy_score(y_true_bin, y_pred_bin)),
            "MacroF1": float(f1_score(y_true_bin, y_pred_bin,
                                       average="macro", zero_division=0)),
            "edges": f"<{edges[0]:.2f} | [{edges[0]:.2f},{edges[1]:.2f}) | >={edges[1]:.2f}",
        },
    }


# ---------------------------------------------------------------------------
# Sonuçları topla & yaz
# ---------------------------------------------------------------------------
STRATEGIES: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], dict]] = {
    "residual_pls_lgbm":     strategy_residual_pls_lgbm,
    "stacking_ga":           strategy_stacking_ga,
    "boosting_ga":           strategy_boosting_ga,
    "preproc_ga_pls_v2":     strategy_preproc_ga_pls_v2,
    "multiseed_ga_pls":      strategy_multiseed_ga_pls,
    "regression_to_ordinal": strategy_regression_to_ordinal,
}


def _measure_baseline(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Hazır GA maskesi ile GA+PLS'in iki metriğini ölç → adil baseline."""
    Xs = X[:, mask]
    n_comp = _pls_n_components(Xs.shape[1], len(y))
    return _eval_dual(lambda: _scaled_pls(n_comp), Xs, y)


def _build_table(all_results: dict[str, dict],
                 baseline_fold: float, baseline_oof: float) -> pd.DataFrame:
    rows = []
    for strat, sub in all_results.items():
        for model_name, m in sub.items():
            row = {"strategy": strat, "model": model_name}
            row.update(m)
            if "R2" in m and not np.isnan(m.get("R2", float("nan"))):
                row["delta_OOF_vs_baseline"] = round(m["R2"] - baseline_oof, 4)
            if "R2_fold_mean" in m and not np.isnan(m.get("R2_fold_mean", float("nan"))):
                row["delta_FOLD_vs_baseline"] = round(m["R2_fold_mean"] - baseline_fold, 4)
            rows.append(row)
    return pd.DataFrame(rows)


def _save_outputs(df: pd.DataFrame, all_results: dict[str, dict],
                  baseline_fold: float, baseline_oof: float,
                  elapsed: float, cfg_source: Path | None) -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = STAGE_DIR / "comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    json_path = STAGE_DIR / "comparison.json"
    payload = {
        "baseline_fold_mean_R2": baseline_fold,
        "baseline_OOF_R2": baseline_oof,
        "results": all_results,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "Flavonol Kombinasyon Stratejileri (v2) — Özet",
        "=" * 72,
        f"Hazır GA maskesi (12_ga_feature_selection):",
        f"  R²_fold_mean (GA modülü-uyumlu) = {baseline_fold:+.3f}",
        f"  R²_OOF (cross_val_predict)      = {baseline_oof:+.3f}",
        f"Toplam süre: {elapsed:.1f} sn ({elapsed/60:.1f} dk)",
        "",
    ]
    if "R2_fold_mean" in df.columns:
        srt = df.dropna(subset=["R2_fold_mean"]).sort_values("R2_fold_mean", ascending=False)
        lines.append("R²_fold_mean ile sıralı (ilk 25):")
        lines.append("-" * 72)
        for _, r in srt.head(25).iterrows():
            d_f = r.get("delta_FOLD_vs_baseline", float("nan"))
            d_o = r.get("delta_OOF_vs_baseline", float("nan"))
            d_f_str = f"{d_f:+.3f}" if pd.notna(d_f) else "  n/a"
            d_o_str = f"{d_o:+.3f}" if pd.notna(d_o) else "  n/a"
            lines.append(
                f"  R²f={r['R2_fold_mean']:+.3f} (Δf={d_f_str}) | "
                f"R²o={r.get('R2', float('nan')):+.3f} (Δo={d_o_str}) | "
                f"RMSE={r.get('RMSE', float('nan')):.3f} | "
                f"{r['strategy']:>22} :: {r['model']}"
            )
    lines += [
        "",
        f"Tüm sonuçlar : {csv_path}",
        f"JSON         : {json_path}",
        "",
        "Notasyon: R²f=fold-mean (GA modülüyle aynı), R²o=OOF (konservatif)",
    ]
    summary_text = "\n".join(lines) + "\n"
    (STAGE_DIR / "summary.txt").write_text(summary_text, encoding="utf-8-sig")

    # Bar chart — fold-mean R²
    if "R2_fold_mean" in df.columns and df["R2_fold_mean"].notna().any():
        plot_df = df.dropna(subset=["R2_fold_mean"]).copy()
        plot_df["label"] = plot_df["strategy"] + " :: " + plot_df["model"]
        plot_df = plot_df.sort_values("R2_fold_mean")
        fig, ax = plt.subplots(1, 1, figsize=(11, max(4, 0.32 * len(plot_df))))
        colors = ["seagreen" if v >= baseline_fold else "steelblue"
                  for v in plot_df["R2_fold_mean"]]
        ax.barh(plot_df["label"], plot_df["R2_fold_mean"], color=colors, edgecolor="k")
        ax.axvline(baseline_fold, color="red", linestyle="--", linewidth=1.2,
                   label=f"baseline R²f={baseline_fold:.3f}")
        ax.axvline(baseline_oof, color="orange", linestyle=":", linewidth=1.2,
                   label=f"baseline R²o={baseline_oof:.3f}")
        ax.set_xlabel("R² (fold-mean, 5-fold CV)")
        ax.set_title("Flavonol — kombinasyon stratejileri (v2)",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        fig.savefig(STAGE_DIR / "comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    paths.write_source_marker(
        STAGE_DIR,
        producer="src/m06_models/flavonol_combos.py",
        config_source=cfg_source,
    )


# ---------------------------------------------------------------------------
# Pipeline & CLI giriş noktaları
# ---------------------------------------------------------------------------
def run(cfg=None, strategies: list[str] | None = None) -> Path:
    chosen = list(STRATEGIES) if not strategies else [s for s in strategies if s in STRATEGIES]
    if not chosen:
        raise ValueError(f"Geçerli strateji yok. Seçenekler: {list(STRATEGIES)}")

    log.info("Flavonol kombinasyonları (v2) başlıyor: %s", chosen)
    X, y, mask = _load_data()

    log.info("Adil baseline ölçülüyor (hazır GA maskesi + PLS)...")
    base = _measure_baseline(X, y, mask)
    log.info("  R²_fold_mean=%.3f  |  R²_OOF=%.3f", base["R2_fold_mean"], base["R2"])

    t0 = time.time()
    all_results: dict[str, dict] = {}
    for name in chosen:
        log.info("=== Strateji: %s ===", name)
        ts = time.time()
        try:
            all_results[name] = STRATEGIES[name](X, y, mask)
            log.info("  bitti (%.1fs)", time.time() - ts)
        except Exception as exc:
            log.exception("  hata: %s", exc)
            all_results[name] = {"hata": {"R2": float("nan"), "R2_fold_mean": float("nan")}}

    elapsed = time.time() - t0

    df = _build_table(all_results, base["R2_fold_mean"], base["R2"])
    cfg_source = cfg.source if cfg is not None else None
    _save_outputs(df, all_results, base["R2_fold_mean"], base["R2"], elapsed, cfg_source)

    log.info("Tamamlandı: %d satır → %s", len(df), STAGE_DIR / "comparison.csv")
    print("\n" + (STAGE_DIR / "summary.txt").read_text(encoding="utf-8-sig"))
    return STAGE_DIR / "summary.txt"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flavonol kombinasyon stratejileri (v2)")
    p.add_argument("--strategies", nargs="*", default=None,
                   choices=list(STRATEGIES),
                   help="Çalıştırılacak stratejiler (varsayılan: hepsi)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s | %(message)s",
                         datefmt="%H:%M:%S")
    args = _parse_args(argv)
    run(cfg=None, strategies=args.strategies)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else None)
