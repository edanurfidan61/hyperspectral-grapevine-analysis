"""GÖREV 10 — Kombine final deneyleri (16_final_combos).

Önceki çalıştırmaların artefaktlarından besleyerek 5 spesifik pipeline'ı
yan-yana karşılaştırır:

| Kod | Pipeline                              | Veri                | HP                         |
|-----|---------------------------------------|---------------------|----------------------------|
| F1  | GA[PLS] + PLS (default)               | GA[flavonol_pls]    | PLS varsayılan             |
| F2  | GA[PLS] + Tuned PLS                   | GA[flavonol_pls]    | 06b/pls/flavonol fine_best |
| F3  | GA[LightGBM] + Tuned LightGBM         | GA[flavonol_lgbm]   | 06b/lightgbm/flavonol fine |
| F4  | 1. türev + GA[PLS] + Tuned PLS        | d1snv_R* ∩ GA mask  | 06b/pls/flavonol fine_best |
| F5  | Sadece indeks + Tuned RF (stres)      | indeks kolonları    | 07b/random_forest/stress   |

Çıktı (``outputs/16_final_combos/``):
- ``<F>/metrics.json`` + ``scatter.png`` (reg) veya ``confusion.png`` (cls)
- ``comparison.csv`` — tüm F'ler + ilgili baseline ile karşılaştırma
- ``summary.md`` — kısa Türkçe yorum
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import f1_score, r2_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from src.core import paths
from src.core.cv import load_groups, make_cv_splitter
from src.core.logging_setup import get as get_logger
from src.m05_dataset import builder as dataset_builder
from src.m06_models import hp_spaces
from src.m06_models.ablation import _score
from src.m06_models.registry import _build_model, _ensure_models_imported, MODELS_REGRESSION, MODELS_CLASSIFICATION

log = get_logger("m06_models.final_combos")


# ---------------------------------------------------------------------------
# Nested-CV "honest" GA değerlendirmesi
# ---------------------------------------------------------------------------
# Mevcut F1..F4 skorları GA maskesini TÜM veride seçip aynı veride CV ile
# değerlendiriyor → klasik **selection bias**. Jüri savunmasında bu sorun
# olur. Aşağıdaki ``_nested_ga_score`` outer K-fold'un her train'inde
# GA'yı tazeden çalıştırıp val'da skorlar → honest, dürüst skor.
#
# Trade-off: GA outer × n_splits kez çalışır → yavaş. Bu yüzden nested
# modda pop/ngen orta düzeye çekilir (default 100/80). Outer GA (150/150)
# = "production mask"; nested GA = "honest generalization estimate".
def _nested_ga_score(
    *, task: str, model_name: str, X: np.ndarray, y: np.ndarray,
    cfg, groups: np.ndarray | None, override_hp: dict | None,
    ga_model: str, ga_pop: int = 50, ga_ngen: int = 40,
    n_splits: int = 5, seed: int = 42,
    extra_mask: np.ndarray | None = None,
) -> tuple[float, float, float] | None:
    """Outer K-fold: her train'de GA, val'da değerlendirme.

    Returns
    -------
    (mean_score, std_score, mean_n_selected) | None
    """
    # Lazy import: GA modülü ağır (DEAP + multiprocessing)
    from src.m04_features.ga_feature_selection import _run_ga

    cls_map = MODELS_REGRESSION if task == "regression" else MODELS_CLASSIFICATION
    if model_name not in cls_map:
        return None

    valid = ~np.isnan(y) if task == "regression" else np.ones(len(y), dtype=bool)
    Xv, yv = X[valid], y[valid]
    gv = groups[valid] if groups is not None else None

    cv = make_cv_splitter(n_splits=n_splits, task=task, groups=gv,
                          random_state=seed)
    split_kw = {"groups": gv} if gv is not None else {}

    scores: list[float] = []
    n_sel: list[int] = []
    for fold_idx, (tr, va) in enumerate(cv.split(Xv, yv, **split_kw)):
        X_tr, X_va = Xv[tr], Xv[va]
        y_tr, y_va = yv[tr], yv[va]
        g_tr = gv[tr] if gv is not None else None

        # F4 için extra_mask = 1.türev kolonları; GA bu alt-küme üzerinde koşar
        if extra_mask is not None:
            X_tr_sub, X_va_sub = X_tr[:, extra_mask], X_va[:, extra_mask]
        else:
            X_tr_sub, X_va_sub = X_tr, X_va

        try:
            ga_mask, _, _ = _run_ga(
                X_tr_sub, y_tr, model=ga_model,
                pop_size=ga_pop, n_gen=ga_ngen,
                # n_jobs=-1: GA fitness değerlendirmesini çok çekirdeğe yay.
                # Tek çekirdekte LightGBM nested ~2 saat sürüyordu; pool ile
                # belirgin hızlanma (özellikle F3n LightGBM).
                seed=seed + fold_idx, n_jobs=-1, groups=g_tr,
            )
        except Exception as exc:
            log.warning("Nested GA fold %d HATA: %s", fold_idx, exc)
            return None

        n_sel.append(int(ga_mask.sum()))

        # Final estimator (tuned HP ile)
        est = hp_spaces.make_estimator(task=task, model_name=model_name,
                                       random_state=seed)
        if est is None:
            return None
        if override_hp:
            try:
                est.set_params(**override_hp)
            except Exception:
                pass

        pipe = SkPipeline([
            ("vt", VarianceThreshold(0.0)),
            ("scaler", StandardScaler()),
            ("est", est),
        ])
        try:
            pipe.fit(X_tr_sub[:, ga_mask], y_tr)
            y_pred = pipe.predict(X_va_sub[:, ga_mask])
        except Exception as exc:
            log.warning("Nested fold %d fit/predict HATA: %s", fold_idx, exc)
            return None

        if task == "regression":
            scores.append(float(r2_score(y_va, y_pred)))
        else:
            scores.append(float(f1_score(y_va, y_pred, average="macro")))

    if not scores:
        return None
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(n_sel))


def _load_ga_mask(ga_subdir: str) -> np.ndarray | None:
    """``outputs/12_ga_feature_selection/<subdir>/ga_best_mask.npy`` yükle."""
    p = paths.OUTPUTS_DIR / "12_ga_feature_selection" / ga_subdir / "ga_best_mask.npy"
    if not p.exists():
        log.warning("GA maskesi yok: %s", p)
        return None
    return np.load(p).astype(bool)


def _load_tuned_params(stage: str, model: str, target: str) -> dict:
    """``outputs/<stage>/<model>/<target>/fine_best_params.json``'dan HP oku.

    Yoksa coarse'a düş; o da yoksa boş dict.
    """
    base = paths.OUTPUTS_DIR / stage / model / target
    for fname in ("fine_best_params.json", "best_params.json"):
        p = base / fname
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return dict(data.get("best_params", {}))
    log.warning("Tuned params yok: %s/%s/%s", stage, model, target)
    return {}


def _derivative_mask(feature_names: list[str]) -> np.ndarray:
    """Sadece ``d1snv_R<nm>`` kolonlarını tutan mask."""
    return np.array([n.startswith("d1snv_R") for n in feature_names], dtype=bool)


def _index_mask(feature_names: list[str]) -> np.ndarray:
    """Spektral (snv_R, d1snv_R) DIŞINDA kalan tüm feature'lar = indeks bloğu."""
    return np.array(
        [not (n.startswith("snv_R") or n.startswith("d1snv_R"))
         for n in feature_names],
        dtype=bool,
    )


def _run_combo(
    *, code: str, task: str, model_name: str, target: str, X: np.ndarray, y: np.ndarray,
    cfg, groups: np.ndarray | None, override_hp: dict | None,
    feature_names: list[str], class_names: list[str] | None,
    out_root: Path,
) -> dict[str, Any]:
    """Bir kombinasyonu çalıştır: BaseModel.run(...) ile metrics+plot üret."""
    out_dir = out_root / code
    out_dir.mkdir(parents=True, exist_ok=True)
    cls_map = MODELS_REGRESSION if task == "regression" else MODELS_CLASSIFICATION
    if model_name not in cls_map:
        log.warning("Model yok: %s/%s", task, model_name)
        return {"code": code, "status": "skipped"}

    model = _build_model(cls_map[model_name], cfg)
    if override_hp:
        model.config["hp"] = {**(model.config.get("hp") or {}), **override_hp}

    # nan filter (regresyon)
    valid = ~np.isnan(y) if task == "regression" else np.ones(len(y), dtype=bool)
    Xv, yv = X[valid], y[valid]
    gv = groups[valid] if groups is not None else None

    try:
        res = model.run(Xv, yv, target, out_dir,
                        feature_names=feature_names,
                        class_names=class_names, groups=gv)
        # Ayrıca mean+std skoru _score ile (comparison için tutarlı protokol)
        sc = _score(task, model_name, X, y, cfg, groups=groups, override_hp=override_hp)
        mean, std = (sc if sc is not None else (float("nan"), float("nan")))
        return {
            "code": code, "task": task, "model": model_name, "target": target,
            "n_features": int(X.shape[1]),
            "metric": "R2" if task == "regression" else "Macro_F1",
            "mean": round(mean, 4), "std": round(std, 4),
            "raw_metrics": res["metrics"],
        }
    except Exception as exc:
        log.exception("Combo %s HATA: %s", code, exc)
        return {"code": code, "status": "failed", "error": str(exc)}


def run(cfg) -> Path:
    """16_final_combos aşaması — F1..F5 kombinasyonlarını çalıştır."""
    _ensure_models_imported()
    out_root = paths.stage_dir("16_final_combos")
    paths.write_source_marker(
        out_root, producer="src/m06_models/final_combos.py",
        config_source=cfg.source,
    )

    data = dataset_builder.load()
    X_full = data["X"]
    fn = data["feature_names"]
    group_key = str(cfg.get("cv.group_key", "leaf"))
    groups = load_groups(paths.OUTPUTS_DIR / "01_dataset", key=group_key)
    y_flav = data["y_flav"]
    y_stress = data["y_stress"]
    n_full = X_full.shape[1]

    rows: list[dict] = []

    # --- F1: GA[PLS] + PLS (default HP) ----------------------------------------
    mask_pls = _load_ga_mask("flavonol_pls")
    if mask_pls is not None and mask_pls.size == n_full:
        rows.append(_run_combo(
            code="F1", task="regression", model_name="pls", target="flavonol",
            X=X_full[:, mask_pls], y=y_flav, cfg=cfg, groups=groups,
            override_hp=None, feature_names=[fn[i] for i in np.where(mask_pls)[0]],
            class_names=None, out_root=out_root,
        ))

    # --- F2: GA[PLS] + Tuned PLS ------------------------------------------------
    tuned_pls = _load_tuned_params("06b_regression_tuned", "pls", "flavonol")
    if mask_pls is not None and mask_pls.size == n_full:
        rows.append(_run_combo(
            code="F2", task="regression", model_name="pls", target="flavonol",
            X=X_full[:, mask_pls], y=y_flav, cfg=cfg, groups=groups,
            override_hp=tuned_pls,
            feature_names=[fn[i] for i in np.where(mask_pls)[0]],
            class_names=None, out_root=out_root,
        ))

    # --- F3: GA[LightGBM] + Tuned LightGBM --------------------------------------
    mask_lgbm = _load_ga_mask("flavonol_lightgbm")
    tuned_lgbm = _load_tuned_params("06b_regression_tuned", "lightgbm", "flavonol")
    if mask_lgbm is not None and mask_lgbm.size == n_full:
        rows.append(_run_combo(
            code="F3", task="regression", model_name="lightgbm", target="flavonol",
            X=X_full[:, mask_lgbm], y=y_flav, cfg=cfg, groups=groups,
            override_hp=tuned_lgbm,
            feature_names=[fn[i] for i in np.where(mask_lgbm)[0]],
            class_names=None, out_root=out_root,
        ))

    # --- F4: 1. türev ∩ GA[PLS] + Tuned PLS -------------------------------------
    deriv_mask = _derivative_mask(fn)
    if mask_pls is not None and mask_pls.size == n_full:
        combined = mask_pls & deriv_mask
        if combined.sum() >= 5:                       # PLS'in min component'i için
            rows.append(_run_combo(
                code="F4", task="regression", model_name="pls", target="flavonol",
                X=X_full[:, combined], y=y_flav, cfg=cfg, groups=groups,
                override_hp=tuned_pls,
                feature_names=[fn[i] for i in np.where(combined)[0]],
                class_names=None, out_root=out_root,
            ))
        else:
            log.warning("F4 atlandı: 1.türev ∩ GA[PLS] yetersiz feature (%d)", int(combined.sum()))

    # --- F5: Sadece indeks + Tuned RF (stres) -----------------------------------
    idx_mask = _index_mask(fn)
    tuned_rf_cls = _load_tuned_params("07b_classification_tuned", "random_forest", "stress")
    if idx_mask.sum() > 0:
        rows.append(_run_combo(
            code="F5", task="classification", model_name="random_forest", target="stress",
            X=X_full[:, idx_mask], y=y_stress, cfg=cfg, groups=groups,
            override_hp=tuned_rf_cls,
            feature_names=[fn[i] for i in np.where(idx_mask)[0]],
            class_names=["sağlıklı", "FD", "diğer biyotik", "abiyotik / diğer"],
            out_root=out_root,
        ))

    # ---- F1n..F4n: NESTED-CV honest skorlar ----------------------------------
    # GA'yı her outer fold'un train'inde tazeden çalıştırıp val'da değerlendirir.
    # Selection bias YOK. Yavaş olduğundan pop/ngen küçültülür (50/40).
    nested_cfg = [
        ("F1n", "regression", "pls",      "flavonol", y_flav, None,        "pls",      None),
        ("F2n", "regression", "pls",      "flavonol", y_flav, tuned_pls,   "pls",      None),
        ("F3n", "regression", "lightgbm", "flavonol", y_flav, tuned_lgbm,  "lightgbm", None),
        ("F4n", "regression", "pls",      "flavonol", y_flav, tuned_pls,   "pls",      deriv_mask),
    ]
    seed = int(cfg.get("models.random_state", 42))
    for code, task, mname, tgt, y_arr, hp_, ga_mdl, em in nested_cfg:
        log.info("Nested-CV başlıyor: %s (GA[%s] → %s)", code, ga_mdl, mname)
        res = _nested_ga_score(
            task=task, model_name=mname, X=X_full, y=y_arr, cfg=cfg,
            groups=groups, override_hp=hp_, ga_model=ga_mdl,
            ga_pop=100, ga_ngen=80, n_splits=5, seed=seed,
            extra_mask=em,
        )
        if res is None:
            rows.append({"code": code, "status": "failed",
                         "error": "nested CV başarısız"})
            continue
        mean_n, std_n, n_feat_avg = res
        rows.append({
            "code": code, "task": task, "model": mname, "target": tgt,
            "n_features": round(n_feat_avg, 1),
            "metric": "R2" if task == "regression" else "Macro_F1",
            "mean": round(mean_n, 4), "std": round(std_n, 4),
            "raw_metrics": {"nested_cv": True, "ga_pop": 100, "ga_ngen": 80},
        })
        log.info("%s nested: %.4f ± %.4f (n_feat ort %.1f)",
                 code, mean_n, std_n, n_feat_avg)

    # ---- Bias hesabı: F1..F4 (biased) ↔ F1n..F4n (nested) --------------------
    by_code = {r.get("code"): r for r in rows
               if r.get("status") not in ("failed", "skipped")}
    for biased_code, nested_code in (("F1", "F1n"), ("F2", "F2n"),
                                     ("F3", "F3n"), ("F4", "F4n")):
        b, n = by_code.get(biased_code), by_code.get(nested_code)
        if b and n:
            n["bias_vs_biased"] = round(b["mean"] - n["mean"], 4)

    # ---- comparison.csv (mean + std) -----------------------------------------
    if not rows:
        log.warning("16_final_combos: hiç deney çalıştırılamadı")
        return out_root

    df = pd.DataFrame(rows)
    # raw_metrics kolonu CSV'de gürültü → çıkar
    keep_cols = [c for c in df.columns if c != "raw_metrics"]
    df_csv = df[keep_cols]
    df_csv.to_csv(out_root / "comparison.csv", index=False, encoding="utf-8-sig")
    try:
        df_csv.to_excel(out_root / "comparison.xlsx", index=False)
    except Exception as exc:
        log.warning("xlsx atlandı: %s", exc)

    # ---- summary.md ----------------------------------------------------------
    md = ["# 16_final_combos — Özet", "",
          f"Toplam pipeline: **{len(rows)}**", ""]
    md.append("## Sonuçlar (mean ± std)")
    md.append("")
    md.append("| Kod | Görev | Hedef | Model | n_feat | Skor |")
    md.append("|-----|-------|-------|-------|--------|------|")
    for r in rows:
        if r.get("status") in ("failed", "skipped"):
            md.append(f"| {r['code']} | — | — | — | — | _{r.get('status')}_ |")
            continue
        md.append(
            f"| {r['code']} | {r['task']} | {r['target']} | {r['model']} "
            f"| {r['n_features']} | {r['mean']:.4f} ± {r['std']:.4f} |"
        )
    md += [
        "",
        "## Yorum",
        "",
        "- **F1 vs F2**: GA[PLS]+PLS baseline vs GA[PLS]+Tuned PLS — tuning'in",
        "  GA-seçili dar feature kümesinde fark yaratıp yaratmadığı.",
        "- **F3**: GA[LightGBM]+Tuned LightGBM — tuning'de en iyi sınıflandırma",
        "  veren modelin GA ile birleşik performansı.",
        "- **F4**: 1.türev ∩ GA[PLS] + Tuned PLS — proje_ozeti'ndeki tek-başına",
        "  en faydalı blok (1.türev) GA ile birleşince ne olur.",
        "- **F5**: Sadece indeks + Tuned RF — stres sınıflandırmasında indeks",
        "  bloğunun başına yeterli mi.",
        "",
        "## Selection bias uyarısı (F1n..F4n nested-CV)",
        "",
        "F1..F4 satırlarındaki skor, GA maskesini **tüm veride** seçip aynı",
        "veride CV ile değerlendirdiği için **optimistik biased**'tır (klasik",
        "feature-selection bias). Bu yüzden her birinin **dürüst (nested-CV)**",
        "karşılığı F1n..F4n olarak ayrıca raporlanır: her outer fold'un",
        "train'inde GA tazeden çalışır (pop=100, ngen=80), val'da skorlanır.",
        "`bias_vs_biased` kolonu = (biased mean) − (nested mean). Pozitif ve",
        "büyük bir değer, biased skorun ne kadar şişirilmiş olduğunu gösterir.",
        "**Raporda/savunmada F*n skorlarını öne çıkar.**",
    ]
    (out_root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("16_final_combos: %d satır → %s", len(rows), out_root)
    return out_root
