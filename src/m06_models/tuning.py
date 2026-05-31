"""GÖREV 2 — İki aşamalı hiperparametre tuning aşaması.

Pipeline aşaması: ``06b_regression_tuned`` / ``07b_classification_tuned``.

Akış:
    1) Aşama A — Coarse RandomizedSearchCV (``BaseModel.tune_coarse``)
       → ``<out>/<model>/<target>/best_params.json``
    2) Aşama B — Optuna Bayesian fine search (``optuna_tuner.fine_tune``)
       → ``<out>/<model>/<target>/fine_best_params.json``
    3) Tuned HP ile son CV değerlendirmesi (``BaseModel.run``)
       → metrics.json + scatter/confusion + predictions.npz

Eski ``06_regression`` / ``07_classification`` çıktıları DOKUNULMAZ;
karşılaştırma için baseline olarak korunur. Yeni dizinler:
    outputs/06b_regression_tuned/<model>/<target>/
    outputs/07b_classification_tuned/<model>/<target>/
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from src.core import paths
from src.core.cv import load_groups
from src.core.logging_setup import get as get_logger
from src.m05_dataset import builder as dataset_builder
from src.m06_models import hp_spaces
from src.m06_models.registry import (
    CLASSIFICATION_TARGETS,
    MODELS_CLASSIFICATION,
    MODELS_REGRESSION,
    REGRESSION_TARGETS,
    STRESS_CLASS_NAMES,
    _build_model,
    _ensure_models_imported,
    _is_deep,
)
from src.m06_models.utils import pheur_pass_fail

log = get_logger("m06_models.tuning")


def _save_json(path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               default=_json_default), encoding="utf-8")


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _tune_one(
    *,
    task: str, model_name: str, model_cls, cfg,
    X: np.ndarray, y: np.ndarray, groups: np.ndarray | None,
    target_name: str, class_names: list[str] | None,
    out_dir, feature_names,
    coarse_on: bool, fine_on: bool,
    n_iter: int, n_trials: int, timeout: int,
    scoring: str,
) -> dict | None:
    """Tek (model, target) için A+B tuning + tuned değerlendirme."""
    param_dist = hp_spaces.get_param_dist(task, model_name)
    if param_dist is None:
        log.info("HP grid yok, atlanıyor: %s/%s", model_name, target_name)
        return None

    # Tuning-uyumlu sklearn estimator (RidgeCV gibi sarmalayıcıları by-pass)
    base_est = hp_spaces.make_estimator(
        task=task, model_name=model_name,
        random_state=int(cfg.get("models.random_state", 42)),
    )
    if base_est is None:
        log.warning("Estimator fabrikası %s/%s için None döndü, atlanıyor",
                    task, model_name)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    # Tuning sırasında CV/scoring/resampling konfigürasyonu BaseModel'den geliyor
    model = _build_model(model_cls, cfg)

    # Geçerli (NaN-olmayan) alt küme — BaseModel.run ile aynı kural
    valid = ~np.isnan(y) if task == "regression" else np.ones(len(y), dtype=bool)
    Xv, yv = X[valid], y[valid]
    gv = groups[valid] if groups is not None else None

    summary: dict[str, Any] = {"task": task, "model": model_name, "target": target_name}

    # --- Aşama A: Coarse ---
    best_coarse: dict | None = None
    if coarse_on:
        t0 = time.time()
        best_coarse, score_c = model.tune_coarse(
            Xv, yv, groups=gv, param_dist=param_dist,
            n_iter=n_iter, scoring=scoring, estimator=base_est,
        )
        dur = time.time() - t0
        _save_json(out_dir / "best_params.json", {
            "best_params": best_coarse, "cv_score": score_c,
            "scoring": scoring, "n_iter": n_iter, "duration_s": round(dur, 1),
        })
        summary["coarse"] = {"best": best_coarse, "score": score_c, "duration_s": round(dur, 1)}
        log.info("Coarse %s/%s: %s=%.4f (%.1fs)",
                 model_name, target_name, scoring, score_c, dur)

    # --- Aşama B: Fine (Optuna) ---
    best_fine: dict | None = None
    if fine_on and best_coarse is not None:
        from src.m06_models import optuna_tuner

        def _factory(params: dict):
            # her trial için fresh estimator (param-set ile)
            est = hp_spaces.make_estimator(
                task=task, model_name=model_name,
                random_state=int(cfg.get("models.random_state", 42)),
            )
            return est.set_params(**params)

        t0 = time.time()
        best_fine, score_f = optuna_tuner.fine_tune(
            model, Xv, yv, coarse_best=best_coarse, param_dist=param_dist,
            n_trials=n_trials, timeout=timeout, scoring=scoring,
            groups=gv, estimator_factory=_factory,
        )
        dur = time.time() - t0
        _save_json(out_dir / "fine_best_params.json", {
            "best_params": best_fine, "cv_score": score_f,
            "scoring": scoring, "n_trials": n_trials,
            "timeout_s": timeout, "duration_s": round(dur, 1),
        })
        summary["fine"] = {"best": best_fine, "score": score_f, "duration_s": round(dur, 1)}
        log.info("Fine  %s/%s: %s=%.4f (%.1fs)",
                 model_name, target_name, scoring, score_f, dur)

    # --- Tuned HP ile son değerlendirme ---
    final_params = best_fine if best_fine is not None else best_coarse
    if final_params is None:
        return summary

    # model.config["hp"] içine merge ederek BaseModel.run akışını kullan
    model.config["hp"] = {**(model.config.get("hp") or {}), **final_params}
    res = model.run(Xv, yv, target_name, out_dir,
                    feature_names=feature_names,
                    class_names=class_names, groups=gv)
    summary["final_metrics"] = res["metrics"]
    return summary


def run(cfg, task: str) -> dict[str, Any]:
    """Tuning aşaması — task ∈ {"regression", "classification"}."""
    if task not in ("regression", "classification"):
        raise ValueError(f"tuning.run: desteklenmeyen task={task!r}")
    _ensure_models_imported()

    stage = "06b_regression_tuned" if task == "regression" else "07b_classification_tuned"
    stage_dir = paths.stage_dir(stage)
    paths.write_source_marker(
        stage_dir,
        producer=f"src/m06_models/tuning.py · {task}",
        config_source=cfg.source,
    )

    # Konfigürasyon (quick modda çevikleştirilir)
    coarse_on = bool(cfg.get("hp_search.coarse.enabled", True))
    fine_on = bool(cfg.get("hp_search.fine.enabled", True))
    n_iter = int(cfg.get("hp_search.coarse.n_iter", 40))
    n_trials = int(cfg.get("hp_search.fine.n_trials", 50))
    timeout = int(cfg.get("hp_search.fine.timeout_seconds", 600))
    scoring_reg = str(cfg.get("hp_search.coarse.scoring_regression", "r2"))
    scoring_cls = str(cfg.get("hp_search.coarse.scoring_classification", "f1_macro"))
    if bool(cfg.get("quick", False)):
        n_iter = min(8, n_iter)
        n_trials = min(10, n_trials)
        timeout = min(60, timeout)
        log.info("QUICK modu: n_iter=%d, n_trials=%d, timeout=%ds",
                 n_iter, n_trials, timeout)

    data = dataset_builder.load()
    X = data["X"]
    feature_names = data["feature_names"]
    group_key = str(cfg.get("cv.group_key", "leaf"))
    groups = load_groups(paths.OUTPUTS_DIR / "01_dataset", key=group_key)

    summaries: list[dict] = []

    if task == "regression":
        scoring = scoring_reg
        for target_name, ykey in REGRESSION_TARGETS.items():
            y = data[ykey]
            for model_name, model_cls in MODELS_REGRESSION.items():
                if _is_deep(model_cls):
                    continue
                try:
                    s = _tune_one(
                        task=task, model_name=model_name, model_cls=model_cls, cfg=cfg,
                        X=X, y=y, groups=groups,
                        target_name=target_name, class_names=None,
                        out_dir=stage_dir / model_name / target_name,
                        feature_names=feature_names,
                        coarse_on=coarse_on, fine_on=fine_on,
                        n_iter=n_iter, n_trials=n_trials, timeout=timeout,
                        scoring=scoring,
                    )
                    if s is not None:
                        summaries.append(s)
                except Exception as exc:
                    log.exception("Tuning HATA %s/%s: %s", model_name, target_name, exc)
    else:  # classification
        scoring = scoring_cls
        # Çok-sınıflı stres
        for target_name, ykey in CLASSIFICATION_TARGETS.items():
            y = data[ykey]
            for model_name, model_cls in MODELS_CLASSIFICATION.items():
                if _is_deep(model_cls):
                    continue
                if model_name == "pheur":
                    continue  # ayrıca aşağıda binary olarak işleniyor
                try:
                    s = _tune_one(
                        task=task, model_name=model_name, model_cls=model_cls, cfg=cfg,
                        X=X, y=y, groups=groups,
                        target_name=target_name, class_names=STRESS_CLASS_NAMES,
                        out_dir=stage_dir / model_name / target_name,
                        feature_names=feature_names,
                        coarse_on=coarse_on, fine_on=fine_on,
                        n_iter=n_iter, n_trials=n_trials, timeout=timeout,
                        scoring=scoring,
                    )
                    if s is not None:
                        summaries.append(s)
                except Exception as exc:
                    log.exception("Tuning HATA %s/%s: %s", model_name, target_name, exc)

        # Ph.Eur. binary (SVC) — y_flav eşikten
        if "pheur" in MODELS_CLASSIFICATION:
            try:
                threshold = float(cfg.get("targets.pheur_flavonol_threshold", 3.5))
                y_bin = pheur_pass_fail(data["y_flav"], threshold=threshold)
                s = _tune_one(
                    task=task, model_name="pheur",
                    model_cls=MODELS_CLASSIFICATION["pheur"], cfg=cfg,
                    X=X, y=y_bin, groups=groups,
                    target_name="flavonol_pheur",
                    class_names=["FAIL (<eşik)", "PASS (>=eşik)"],
                    out_dir=stage_dir / "pheur_binary" / "flavonol_pheur",
                    feature_names=feature_names,
                    coarse_on=coarse_on, fine_on=fine_on,
                    n_iter=n_iter, n_trials=n_trials, timeout=timeout,
                    scoring=scoring,
                )
                if s is not None:
                    summaries.append(s)
            except Exception as exc:
                log.exception("Tuning HATA pheur_binary: %s", exc)

    # Toplu özet
    _save_json(stage_dir / "tuning_summary.json", {
        "task": task, "n_runs": len(summaries), "results": summaries,
    })
    log.info("%s tuning tamamlandı: %d run", task, len(summaries))
    return {"n_runs": len(summaries)}
