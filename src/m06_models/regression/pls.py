"""Partial Least Squares Regression — spektroskopinin altın standardı.

Yüksek boyutlu, kolineer spektral özelliklerde Ridge'den genelde üstündür.
``n_components`` özellik sayısı ve örnek sayısının yarısını geçmemeli.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression as _SKPLS

from src.m06_models.base import BaseModel
from src.m06_models.registry import register


class _PLS1D(_SKPLS):
    """Sağlamlaştırılmış PLS.

    1. ``predict`` çıktısını 1D yapar — BaseModel metric pipeline'ı için gerekli.
    2. ``fit``: ``n_components`` veri rankından büyükse veya kolineer veride
       (ör. dar 1.türev ∩ GA alt-kümesi) PLS deflasyonu dejenere olup
       "A has a NaN entry" hatası verirse, ``n_components``'i kademeli azaltıp
       yeniden dener. Böylece çökme yerine en yüksek geçerli bileşenle fit olur.
    """

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_samples, n_features = X.shape
        requested = int(self.n_components)
        # Rank tahmini (NaN/inf'i 0'a sabitle ki matrix_rank patlamasın).
        X_finite = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        eff_rank = int(np.linalg.matrix_rank(X_finite)) if n_features > 0 else 1
        hard_cap = max(1, min(requested, n_features, n_samples - 1, eff_rank))

        last_err: Exception | None = None
        for nc in range(hard_cap, 0, -1):
            self.n_components = nc
            try:
                return super().fit(X, y)
            except Exception as exc:        # ValueError: A has a NaN entry
                last_err = exc
        self.n_components = requested
        raise last_err if last_err is not None else RuntimeError("PLS fit başarısız")

    def predict(self, X, copy=True):
        out = super().predict(X, copy=copy)
        return np.asarray(out).ravel()


@register("regression", "pls")
class PLSReg(BaseModel):
    requires_scaling = True

    def _build_estimator(self) -> Any:
        hp = self.config.get("hp", {}) or {}
        n_components = int(hp.get("n_components", 10))
        max_iter = int(hp.get("max_iter", 500))
        return _PLS1D(n_components=n_components, max_iter=max_iter, scale=False)
