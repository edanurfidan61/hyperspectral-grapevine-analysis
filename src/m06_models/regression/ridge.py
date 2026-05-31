"""Ridge regression model adapter (kayıt: 'ridge')."""

from __future__ import annotations

from src.m06_models.registry import register
from src.m06_models.base import BaseModel

try:
    from sklearn.linear_model import RidgeCV
except Exception:  # pragma: no cover - sklearn should be available in normal dev
    RidgeCV = None


@register("regression", "ridge")
class RidgeReg(BaseModel):
    """Ridge regression using sklearn's RidgeCV (grid of alphas)."""

    def _build_estimator(self):
        hp = self.config.get("hp", {}) or {}
        alphas = hp.get("alphas", [1.0])
        if RidgeCV is None:
            raise RuntimeError("sklearn not available: RidgeCV")
        return RidgeCV(alphas=alphas, cv=self.cv)
