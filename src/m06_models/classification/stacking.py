"""Stacking classifier (kayıt: 'stacking')."""

from __future__ import annotations

from src.m06_models.registry import register
from src.m06_models.base import BaseModel

from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


@register("classification", "stacking")
class StackingCls(BaseModel):
    task = "classification"

    def _build_estimator(self):
        hp = self.config.get("hp", {}) or {}
        base = hp.get("base", ["random_forest", "lightgbm"])

        estimators = []
        for name in base:
            if name == "random_forest":
                estimators.append(("rf", RandomForestClassifier(n_estimators=100, random_state=self.random_state)))
            elif name == "lightgbm" and LGBMClassifier is not None:
                estimators.append(("lgbm", LGBMClassifier(n_estimators=100, random_state=self.random_state, verbose=-1, verbosity=-1, force_col_wise=True)))

        final = LogisticRegression(max_iter=1000)
        return StackingClassifier(estimators=estimators, final_estimator=final, n_jobs=-1)
