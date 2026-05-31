"""LightGBM classifier (kayıt: 'lightgbm'). Falls back to RF if missing."""

from __future__ import annotations

from src.m06_models.registry import register
from src.m06_models.base import BaseModel

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


@register("classification", "lightgbm")
class LGBMCls(BaseModel):
    task = "classification"

    def _build_estimator(self):
        hp = self.config.get("hp", {}) or {}
        if LGBMClassifier is None:
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(n_estimators=int(hp.get("n_estimators", 100)), random_state=self.random_state, n_jobs=-1)
        return LGBMClassifier(
            n_estimators=int(hp.get("n_estimators", 100)),
            learning_rate=float(hp.get("learning_rate", 0.05)),
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,                # Python tarafı
            verbosity=-1,              # C++ tarafı (stderr)
            force_col_wise=True,
        )
