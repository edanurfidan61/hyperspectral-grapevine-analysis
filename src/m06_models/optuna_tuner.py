"""GÖREV 2 / Aşama B — Optuna ile dar-alan fine tuning.

Coarse'tan (RandomizedSearchCV) gelen ``best_params``'ı merkez alıp etrafında
~50 trial Bayesian search yapar:
    - Numerik parametreler: ``best ± %50``  (log-scale uygun olduğunda log uniform)
    - Kategorik parametreler: coarse grid'inden yeniden örnekle (best dahil)

Sampler: ``TPESampler``; Pruner: ``MedianPruner``. Trial başına ortalama CV
skoru döner; intermediate raporlama yok (tek skor → pruner pratik etkisi
sınırlı, plan tarafından istendiği için yine de tanımlanır).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.logging_setup import get as get_logger

log = get_logger("m06_models.optuna_tuner")


def _suggest(trial, name: str, coarse_choices: list, coarse_best: Any) -> Any:
    """Tek bir parametre için Optuna önerisi.

    Numerik ise best ± %50; kategorik ise eski grid + best birleşimi.
    """
    is_numeric = all(isinstance(c, (int, float)) for c in coarse_choices if c is not None)
    has_none = any(c is None for c in coarse_choices)

    if not is_numeric or has_none:
        # Kategorik veya None içeren: orijinal grid + best ile suggest_categorical
        choices = list({*coarse_choices, coarse_best})
        return trial.suggest_categorical(name, choices)

    # Numerik: dar alan
    if coarse_best is None:
        return trial.suggest_categorical(name, coarse_choices)
    # Coarse grid'in min/max'ı (örn. feature_fraction için [0.7, 1.0]); fine
    # pencere bu sınırların dışına ÇIKMAMALI (LightGBM feature_fraction > 1.0
    # için fatal verir).
    grid_nums = [c for c in coarse_choices if isinstance(c, (int, float))
                 and not isinstance(c, bool)]
    grid_lo = min(grid_nums) if grid_nums else None
    grid_hi = max(grid_nums) if grid_nums else None

    if isinstance(coarse_best, int) and not isinstance(coarse_best, bool):
        lo = max(1, int(round(coarse_best * 0.5)))
        hi = max(lo + 1, int(round(coarse_best * 1.5)) + 1)
        if grid_lo is not None:
            lo = max(lo, int(grid_lo))
        if grid_hi is not None:
            hi = min(hi, int(grid_hi))
        if hi <= lo:
            hi = lo + 1
        return trial.suggest_int(name, lo, hi)
    # float
    v = float(coarse_best)
    if v <= 0:
        # log uniform için pozitiflik şart; küçük epsilonla pencere
        lo, hi = max(1e-9, abs(v) * 0.5 + 1e-9), abs(v) * 1.5 + 1e-6
        return trial.suggest_float(name, lo, hi)
    lo, hi = v * 0.5, v * 1.5
    if grid_lo is not None:
        lo = max(lo, float(grid_lo))
    if grid_hi is not None:
        hi = min(hi, float(grid_hi))
    if hi <= lo:
        hi = lo * 1.01 + 1e-9
    return trial.suggest_float(name, lo, hi, log=True)


def fine_tune(
    model,                         # BaseModel örneği
    X: np.ndarray,
    y: np.ndarray,
    *,
    coarse_best: dict,
    param_dist: dict,
    n_trials: int = 50,
    timeout: int = 600,
    scoring: str | None = None,
    groups: np.ndarray | None = None,
    estimator_factory=None,
) -> tuple[dict, float]:
    """Coarse best etrafında Optuna Bayesian search.

    Parameters
    ----------
    model
        ``BaseModel`` örneği — CV/scaling/resampling konfigürasyonu okunur.
    coarse_best
        Aşama A'dan gelen en iyi parametre seti (clean, prefix'siz).
    param_dist
        Coarse grid (numerik vs kategorik tanımak için referans).
    estimator_factory
        ``callable(params) -> sklearn estimator``. Verilmezse
        ``model._build_estimator`` kullanılır (config["hp"] güncellenip).

    Returns
    -------
    (best_params, best_score)
    """
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    from sklearn.model_selection import cross_val_score
    from src.core.cv import make_cv_splitter

    # CV'yi bir kez kur (her trial yeniden örnekleme yapmasın)
    cv = make_cv_splitter(
        n_splits=model.cv, task=model.task,
        groups=groups, random_state=model.random_state,
        stratify_regression=model.stratify_regression,
        n_bins=model.regression_n_bins,
    )

    # Pipeline iskeleti — coarse ile aynı (leakage'sız, SMOTE varsa fold-içi)
    from sklearn.preprocessing import StandardScaler

    use_resample = (
        model.task == "classification"
        and getattr(model, "resampling_enabled", False)
        and getattr(model, "resampling_method", "none") != "none"
    )

    def _build_pipe(params: dict):
        if estimator_factory is not None:
            est = estimator_factory(params)
        else:
            model.config["hp"] = {**model.config.get("hp", {}), **params}
            est = model._build_estimator()
        steps = []
        if model.requires_scaling:
            steps.append(("scaler", StandardScaler()))
        if use_resample:
            from imblearn.pipeline import Pipeline as ImbPipeline
            from src.m06_models.utils import SafeResampler
            steps.append(("resampler", SafeResampler(
                method=model.resampling_method, random_state=model.random_state,
            )))
            steps.append(("est", est))
            return ImbPipeline(steps)
        from sklearn.pipeline import Pipeline as SkPipeline
        steps.append(("est", est))
        return SkPipeline(steps)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            k: _suggest(trial, k, choices, coarse_best.get(k))
            for k, choices in param_dist.items()
        }
        pipe = _build_pipe(params)
        try:
            scores = cross_val_score(
                pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1,
                groups=groups if groups is not None else None,
                error_score="raise",
            )
        except Exception as exc:
            # Bazı param kombinasyonları küçük dataset'te kırılabilir → bu trial'ı budama
            raise optuna.TrialPruned(str(exc))
        return float(np.mean(scores))

    sampler = TPESampler(seed=model.random_state)
    pruner = MedianPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    log.info("Optuna fine: n_trials=%d, timeout=%ds, scoring=%s",
             n_trials, timeout, scoring)
    study.optimize(objective, n_trials=int(n_trials), timeout=int(timeout),
                   show_progress_bar=False)

    return dict(study.best_params), float(study.best_value)
