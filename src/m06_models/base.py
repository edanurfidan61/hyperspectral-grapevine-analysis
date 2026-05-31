"""Tüm modeller için ortak arayüz: ``BaseModel``.

Klasik (sklearn) ve derin (PyTorch) modeller bu sınıftan miras alır. Eski projedeki
24 modelin %70'i kopyala-yapıştır olan eğitim/değerlendirme şablonu burada toplanır;
her concrete model 30-50 satıra iner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.core.cv import make_cv_splitter
from src.core.logging_setup import get as get_logger
from src.m06_models.utils import (
    apply_smote_tomek,
    classify_metrics,
    plot_confusion,
    plot_regression,
    plot_residuals,
    regression_metrics,
    save_metrics_json,
)


class BaseModel(ABC):
    """Tüm modeller için ortak arayüz.

    Bir alt sınıf yapması gereken minimum: ``name``, ``task`` ve ``_build_estimator()``.
    """

    name: str = ""
    task: Literal["regression", "classification"] = "regression"
    requires_scaling: bool = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.cv: int = int(self.config.get("cv", 5))
        self.random_state: int = int(self.config.get("random_state", 42))
        self.scaler: StandardScaler | None = None
        self.estimator: Any | None = None
        self.log = get_logger(f"m06_models.{self.task}.{self.name or 'unknown'}")
        # Hedef log-transform aktiflenecek hedef adları (ör. ["flavonol"])
        self.log_transform_targets: list[str] = list(
            self.config.get("log_transform_targets", []) or []
        )
        # Regresyonda quantile-bin tabanlı stratified CV (continuous binning)
        self.stratify_regression: bool = bool(
            self.config.get("stratify_regression", False)
        )
        self.regression_n_bins: int = int(
            self.config.get("regression_n_bins", 5)
        )
        # GÖREV 1: sınıflandırma için resampling (SMOTE / SMOTE+Tomek)
        # Açıksa cross_val_predict yerine manuel fold döngüsü kullanılır
        # (scaler + resampling SADECE train fold'una uygulanır).
        self.resampling_enabled: bool = bool(
            self.config.get("resampling_enabled", False)
        )
        self.resampling_method: str = str(
            self.config.get("resampling_method", "none")
        )

    # ---- abstract -----------------------------------------------------------
    @abstractmethod
    def _build_estimator(self) -> Any:
        """Konkret modelin sklearn-uyumlu (fit/predict'li) tahmincisini üretir."""

    # ---- ortak fit/predict --------------------------------------------------
    def _scale_fit_transform(self, X: np.ndarray) -> np.ndarray:
        if not self.requires_scaling:
            return X
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X)

    def _scale_transform(self, X: np.ndarray) -> np.ndarray:
        if not self.requires_scaling or self.scaler is None:
            return X
        return self.scaler.transform(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        Xs = self._scale_fit_transform(X)
        self.estimator = self._build_estimator()
        self.estimator.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Model henüz fit edilmedi.")
        return self.estimator.predict(self._scale_transform(X))

    # ---- cross-validation tahminleri ---------------------------------------
    def cross_val_predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        """5-fold CV tahmini. ``groups`` verilirse leakage-koruyucu split.

        NOT: Bu yol scaler'ı CV'den önce tüm X üzerine fit eder (hafif leakage).
        Sınıflandırma + resampling açıkken ``_cv_predict_resampled`` kullanılır
        ve scaler de fold-içi olur — orada leakage tamamen kapanır.
        """
        Xs = self._scale_fit_transform(X)
        est = self._build_estimator()
        cv = make_cv_splitter(
            n_splits=self.cv, task=self.task,
            groups=groups, random_state=self.random_state,
            stratify_regression=self.stratify_regression,
            n_bins=self.regression_n_bins,
        )
        if groups is not None:
            return cross_val_predict(est, Xs, y, cv=cv, groups=groups)
        return cross_val_predict(est, Xs, y, cv=cv)

    # ---- Manuel fold + train-only resampling (GÖREV 1) ---------------------
    def _cv_predict_resampled(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sınıflandırma için manuel CV: scaler + SMOTE SADECE train fold'una.

        Her fold için:
            1) StandardScaler.fit_transform(X_train)   ← train-only fit
            2) apply_smote_tomek(X_train_s, y_train)   ← train-only resample
            3) scaler.transform(X_val)
            4) estimator.fit(X_resampled, y_resampled)
            5) predict(X_val) → y_pred[val_idx]
        Validation'a NE scaler.fit NE resampling uygulanır (leakage yasak).
        """
        cv = make_cv_splitter(
            n_splits=self.cv, task=self.task,
            groups=groups, random_state=self.random_state,
            stratify_regression=self.stratify_regression,
            n_bins=self.regression_n_bins,
        )
        # sklearn CV API: groups verildiyse split(X, y, groups), aksi halde split(X, y)
        splitter = cv.split(X, y, groups=groups) if groups is not None else cv.split(X, y)

        y_pred = np.empty(len(y), dtype=y.dtype)
        for fold_idx, (tr, va) in enumerate(splitter, start=1):
            # 1+3) Scaling (fold-içi)
            if self.requires_scaling:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X[tr])
                X_va_s = scaler.transform(X[va])
            else:
                X_tr_s, X_va_s = X[tr], X[va]
            # 2) Resampling (train-only)
            X_res, y_res = apply_smote_tomek(
                X_tr_s, y[tr],
                method=self.resampling_method,
                random_state=self.random_state,
            )
            # 4+5) Fit & predict
            est = self._build_estimator()
            est.fit(X_res, y_res)
            y_pred[va] = est.predict(X_va_s)
            self.log.debug(
                "fold %d: train=%d→%d, val=%d", fold_idx, len(tr), len(X_res), len(va),
            )
        return y_pred

    # ---- Coarse hiperparametre araması (GÖREV 2 — Aşama A) -----------------
    def tune_coarse(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        param_dist: dict,
        n_iter: int = 40,
        scoring: str | None = None,
        groups: np.ndarray | None = None,
        estimator: object | None = None,
    ) -> tuple[dict, float]:
        """RandomizedSearchCV ile coarse hiperparametre araması.

        Inner CV ``make_cv_splitter`` ile group-aware oluşturulur. Sınıflandırma
        + resampling açıkken estimator imblearn Pipeline (Scaler + SafeResampler
        + estimator) olarak sarmalanır; SMOTE her zaman fold-içi (leakage'sız).

        ``estimator`` parametresi verilmezse ``self._build_estimator()`` kullanılır
        — ancak tuning tarafında çoğunlukla ``hp_spaces.make_estimator`` çıktısı
        tercih edilir (RidgeCV gibi nested CV yapanları devre dışı bırakmak için).
        """
        from sklearn.model_selection import RandomizedSearchCV

        base_est = estimator if estimator is not None else self._build_estimator()
        cv = make_cv_splitter(
            n_splits=self.cv, task=self.task,
            groups=groups, random_state=self.random_state,
            stratify_regression=self.stratify_regression,
            n_bins=self.regression_n_bins,
        )

        # Pipeline'ı kur — scaling + (varsa) resampling + estimator
        from sklearn.preprocessing import StandardScaler
        steps = []
        if self.requires_scaling:
            steps.append(("scaler", StandardScaler()))
        use_resample = (
            self.task == "classification"
            and self.resampling_enabled
            and self.resampling_method != "none"
        )
        if use_resample:
            from imblearn.pipeline import Pipeline as ImbPipeline
            from src.m06_models.utils import SafeResampler
            steps.append(("resampler", SafeResampler(
                method=self.resampling_method, random_state=self.random_state,
            )))
            steps.append(("est", base_est))
            pipe = ImbPipeline(steps)
        else:
            from sklearn.pipeline import Pipeline as SkPipeline
            steps.append(("est", base_est))
            pipe = SkPipeline(steps)

        # Param adlarını pipeline prefix'iyle eşle
        param_grid = {f"est__{k}": v for k, v in param_dist.items()}

        search = RandomizedSearchCV(
            pipe, param_distributions=param_grid,
            n_iter=int(n_iter), scoring=scoring, cv=cv,
            n_jobs=-1, random_state=self.random_state,
            refit=True, error_score="raise",
        )
        if groups is not None:
            search.fit(X, y, groups=groups)
        else:
            search.fit(X, y)

        # Pipeline prefix'ini temizleyerek best_params döndür
        best = {k.removeprefix("est__"): v for k, v in search.best_params_.items()}
        return best, float(search.best_score_)

    # ---- yüksek seviye: fit + cross_val + kaydet ---------------------------
    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_name: str,
        out_dir: Path,
        feature_names: list[str] | None = None,
        class_names: list[str] | None = None,
        groups: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Modeli CV ile değerlendir, metrikleri ve grafikleri ``out_dir`` altına yaz.

        ``out_dir`` genellikle ``outputs/04_regression/<model_name>/<target_name>/``
        veya ``outputs/05_classification/<model_name>/`` olur.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        valid = ~np.isnan(y) if self.task == "regression" else np.ones(len(y), dtype=bool)
        if not valid.all():
            self.log.info("%d NaN örneği atlanıyor (%s)", int((~valid).sum()), target_name)
        Xv, yv = X[valid], y[valid]
        gv = groups[valid] if groups is not None else None

        # Flavonol gibi çarpık dağılımlı hedefler için log1p dönüşümü:
        # eğitimde y_log = log1p(y), tahmin sonrası geri al → metrikler orijinal ölçekte.
        use_log = (
            self.task == "regression"
            and target_name in self.log_transform_targets
        )
        if use_log:
            self.log.info("Hedef log-transform: %s (log1p)", target_name)
            yv_train = np.log1p(np.maximum(yv, -1.0 + 1e-9))
        else:
            yv_train = yv

        # GÖREV 1: classification + resampling enabled → manuel fold döngüsü
        use_resampling = (
            self.task == "classification"
            and self.resampling_enabled
            and self.resampling_method != "none"
        )
        cv_mode = (
            f" | resample={self.resampling_method}" if use_resampling
            else ""
        )
        self.log.info("CV başlıyor: %s | n=%d | model=%s%s%s",
                      target_name, len(yv), self.name,
                      " | group-aware" if gv is not None else "",
                      cv_mode)
        if use_resampling:
            y_pred = self._cv_predict_resampled(Xv, yv_train, groups=gv)
        else:
            y_pred = self.cross_val_predict(Xv, yv_train, groups=gv)
        if use_log:
            y_pred = np.expm1(y_pred)

        if self.task == "regression":
            metrics = regression_metrics(yv, y_pred)
            self.log.info("R²=%.3f RMSE=%.3f RPD=%.2f MAPE=%.1f%%",
                          metrics["R2"], metrics["RMSE"], metrics["RPD"], metrics["MAPE"])
            plot_regression(yv, y_pred, out_dir / "scatter.png",
                            title=f"{self.name} | {target_name}",
                            xlabel=f"Gerçek {target_name}",
                            ylabel=f"Tahmin {target_name}")
            plot_residuals(yv, y_pred, out_dir / "residuals.png",
                           title=f"{self.name} | {target_name}")
        else:
            metrics = classify_metrics(yv, y_pred, class_names=class_names)
            self.log.info("Acc=%.3f BalAcc=%.3f F1=%.3f",
                          metrics["accuracy"], metrics["balanced_accuracy"],
                          metrics["macro_f1"])
            plot_confusion(metrics["confusion_matrix"], out_dir / "confusion.png",
                           class_names=class_names,
                           title=f"{self.name} | {target_name}")

        save_metrics_json(metrics, out_dir / "metrics.json")
        np.savez(out_dir / "predictions.npz", y_true=yv, y_pred=y_pred)
        return {"y_true": yv, "y_pred": y_pred, "metrics": metrics}
