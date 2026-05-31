"""LightGBM regressor adapter (kayıt: 'lightgbm'). Falls back to RF if lightgbm missing."""

from __future__ import annotations

from src.m06_models.registry import register
from src.m06_models.base import BaseModel

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


@register("regression", "lightgbm")
class LGBMReg(BaseModel):
    def _build_estimator(self):
        hp = self.config.get("hp", {}) or {}
        if LGBMRegressor is None:
            # fallback
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(n_estimators=int(hp.get("n_estimators", 100)), random_state=self.random_state, n_jobs=-1)
        return LGBMRegressor(
            n_estimators=int(hp.get("n_estimators", 100)),
            learning_rate=float(hp.get("learning_rate", 0.05)),
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,                # Python tarafı: tüm log seviyelerini kapat
            verbosity=-1,              # C++ tarafı: stderr'e [Info]/[Warning] basmasın
            force_col_wise=True,       # "Auto-choosing col-wise..." mesajını sustur
        )
