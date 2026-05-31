"""HuberRegressor — outlier-dirençli lineer regresyon.

Flavonol gibi çarpık veya outlier'lı hedeflerde Ridge'in MSE-tabanlı kaybına
göre daha sağlam tahmin verir.
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import HuberRegressor

from src.m06_models.base import BaseModel
from src.m06_models.registry import register


@register("regression", "huber")
class HuberReg(BaseModel):
    requires_scaling = True

    def _build_estimator(self) -> Any:
        hp = self.config.get("hp", {}) or {}
        return HuberRegressor(
            epsilon=float(hp.get("epsilon", 1.35)),
            alpha=float(hp.get("alpha", 1e-3)),
            max_iter=int(hp.get("max_iter", 500)),
            tol=float(hp.get("tol", 1e-5)),
        )
