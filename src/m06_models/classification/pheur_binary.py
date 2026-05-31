"""Ph.Eur.-style binary classifier wrapper (kayıt: 'pheur').

SVM-RBF parametreleri konfigürasyondan alınır (varsayılan C=1.0, gamma='scale').
"""

from __future__ import annotations

from src.m06_models.registry import register
from src.m06_models.base import BaseModel

from sklearn.svm import SVC


@register("classification", "pheur")
class PheurBinary(BaseModel):
    task = "classification"
    requires_scaling = True

    def _build_estimator(self):
        hp = self.config.get("hp", {}) or {}
        C = float(hp.get("C", 1.0))
        gamma = hp.get("gamma", "scale")
        kernel = hp.get("kernel", "rbf")
        return SVC(kernel=kernel, C=C, gamma=gamma)
