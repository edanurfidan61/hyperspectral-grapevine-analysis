"""Hiperparametre arama uzayları + tuning-uyumlu sklearn estimator fabrikası.

GÖREV 2 / Aşama A (Coarse RandomizedSearchCV):
    Buradaki ``PARAM_DISTS`` ve ``make_estimator`` ikilisi sklearn'ün
    ``RandomizedSearchCV``'sine doğrudan beslenebilir bir sade estimator üretir.
    Concrete model sınıflarımızın bazıları RidgeCV gibi *kendi içinde tuning yapan*
    estimator döndürdüğü için (nested CV olmasın diye) burada tek-konfigürasyonlu
    sklearn estimator'larını tek noktadan inşa ediyoruz.

GÖREV 2 / Aşama B (Fine Optuna): aynı sözlükler ``optuna_tuner.py`` tarafından
    "dar alan" Bayesian search için referans olarak kullanılır.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Coarse param distributions (RandomizedSearchCV ile uyumlu)
# ---------------------------------------------------------------------------
# REGRESYON
RF_REG: dict[str, list] = {
    "n_estimators": [200, 300, 500, 800],
    "max_depth": [None, 6, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
}

LGBM_REG: dict[str, list] = {
    "n_estimators": [200, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "num_leaves": [15, 31, 63, 127],
    "min_data_in_leaf": [3, 5, 10, 20],
    "feature_fraction": [0.7, 0.8, 0.9, 1.0],
}

PLS: dict[str, list] = {"n_components": list(range(2, 21))}

RIDGE: dict[str, list] = {"alpha": [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]}

HUBER: dict[str, list] = {
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "epsilon": [1.0, 1.35, 1.5, 2.0],
}

# SINIFLANDIRMA
RF_CLS: dict[str, list] = {
    "n_estimators": [200, 300, 500, 800],
    "max_depth": [None, 6, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
    "class_weight": ["balanced", "balanced_subsample"],
}

LGBM_CLS: dict[str, list] = {
    "n_estimators": [200, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "num_leaves": [15, 31, 63, 127],
    "min_data_in_leaf": [3, 5, 10, 20],
    "feature_fraction": [0.7, 0.8, 0.9, 1.0],
    "class_weight": ["balanced", None],
}

# SVM (pheur_binary)
SVC_PARAMS: dict[str, list] = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 1e-3, 1e-2, 0.1, 1],
}


# (task, model_name) → param_dist
_REGISTRY: dict[tuple[str, str], dict[str, list]] = {
    # regression
    ("regression", "ridge"): RIDGE,
    ("regression", "random_forest"): RF_REG,
    ("regression", "lightgbm"): LGBM_REG,
    ("regression", "pls"): PLS,
    ("regression", "huber"): HUBER,
    # classification
    ("classification", "random_forest"): RF_CLS,
    ("classification", "lightgbm"): LGBM_CLS,
    ("classification", "pheur"): SVC_PARAMS,
}


def get_param_dist(task: str, model_name: str) -> dict[str, list] | None:
    """Verili (task, model) için coarse grid — yoksa ``None`` (atlanır)."""
    return _REGISTRY.get((task, model_name))


# ---------------------------------------------------------------------------
# Tuning-uyumlu sklearn estimator fabrikası (RidgeCV gibi nested CV yapanları
# devre dışı bırakır; doğrudan tek-konfigürasyonlu sınıfları döndürür)
# ---------------------------------------------------------------------------
def make_estimator(task: str, model_name: str, random_state: int = 42) -> Any | None:
    """RandomizedSearchCV / Optuna için tek-konfigürasyonlu temel estimator.

    Parametreleri SearchCV ``set_params(**best)`` ile ayarlayacak; burada
    sadece sınıfı ve sabit (model_name'e göre değişmeyen) argümanları belirleriz.
    """
    if task == "regression":
        if model_name == "ridge":
            from sklearn.linear_model import Ridge
            return Ridge(random_state=random_state)
        if model_name == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(random_state=random_state, n_jobs=-1)
        if model_name == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
                return LGBMRegressor(random_state=random_state, n_jobs=-1,
                                     verbose=-1, verbosity=-1, force_col_wise=True)
            except ImportError:
                return None
        if model_name == "pls":
            # Sağlamlaştırılmış PLS: kolineer/dar alt-kümelerde (F4 = 1.türev ∩
            # GA) n_components'i otomatik düşürüp "NaN entry" çökmesini önler.
            from src.m06_models.regression.pls import _PLS1D
            return _PLS1D(max_iter=500, scale=False)
        if model_name == "huber":
            from sklearn.linear_model import HuberRegressor
            return HuberRegressor(max_iter=500)
    elif task == "classification":
        if model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=random_state, n_jobs=-1)
        if model_name == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(random_state=random_state, n_jobs=-1,
                                      verbose=-1, verbosity=-1, force_col_wise=True)
            except ImportError:
                return None
        if model_name == "pheur":
            from sklearn.svm import SVC
            return SVC(kernel="rbf", random_state=random_state)
    return None
