"""Random Forest regressor adapter (kayıt: 'random_forest')."""

from __future__ import annotations

from src.m06_models.registry import register
from src.m06_models.base import BaseModel

from sklearn.ensemble import RandomForestRegressor


@register("regression", "random_forest")
class RFReg(BaseModel):
    def _build_estimator(self):
        hp = self.config.get("hp", {}) or {}
        n_estimators = int(hp.get("n_estimators", 500))
        max_depth = hp.get("max_depth", None)
        return RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=self.random_state, n_jobs=-1
        )
