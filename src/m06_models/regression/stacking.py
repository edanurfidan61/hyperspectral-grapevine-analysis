"""Stacking regressor (kayıt: 'stacking')."""

from __future__ import annotations

from src.m06_models.registry import register
from src.m06_models.base import BaseModel

from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


@register("regression", "stacking")
class StackingReg(BaseModel):
    def _build_estimator(self):
        hp = self.config.get("hp", {}) or {}
        base = hp.get("base", ["ridge", "random_forest", "lightgbm"])

        estimators = []
        for name in base:
            if name == "ridge":
                estimators.append(("ridge", Ridge()))
            elif name == "random_forest":
                estimators.append(("rf", RandomForestRegressor(n_estimators=100, random_state=self.random_state)))
            elif name == "lightgbm" and LGBMRegressor is not None:
                estimators.append(("lgbm", LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1, verbosity=-1, force_col_wise=True)))

        final = Ridge()
        return StackingRegressor(estimators=estimators, final_estimator=final, n_jobs=-1)
