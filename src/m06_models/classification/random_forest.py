"""RandomForest classifier (kayıt: 'random_forest')."""

from __future__ import annotations

from src.m06_models.registry import register
from src.m06_models.base import BaseModel

from sklearn.ensemble import RandomForestClassifier


@register("classification", "random_forest")
class RFCls(BaseModel):
    task = "classification"

    def _build_estimator(self):
        hp = self.config.get("hp", {}) or {}
        n_estimators = int(hp.get("n_estimators", 500))
        class_weight = hp.get("class_weight", None)
        return RandomForestClassifier(
            n_estimators=n_estimators, class_weight=class_weight, random_state=self.random_state, n_jobs=-1
        )
