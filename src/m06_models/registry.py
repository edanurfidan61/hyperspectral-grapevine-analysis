"""Model kayıt defteri ve aşama orkestrasyonu.

Pipeline tek noktadan tüm modelleri dolaşır: ``run_all(cfg, task=...)`` çağrısı
``MODELS_*`` haritasına bakar, her modeli sırasıyla çalıştırır ve
``outputs/<stage>/<model>/<target>/`` altına çıktıları yazar.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from src.core import paths
from src.core.logging_setup import get as get_logger
from src.m05_dataset import builder as dataset_builder
from src.m06_models.base import BaseModel
from src.m06_models.utils import pheur_pass_fail

log = get_logger("m06_models.registry")

# ---- Concrete model sınıflarının kaydı --------------------------------------
# Her modül kendi modelini buraya register eder. Sözlüğün lazy doldurulması
# (import-on-demand) ile pyTorch yokken klasik modeller hâlâ çalışabilir.

MODELS_REGRESSION: dict[str, type[BaseModel]] = {}
MODELS_CLASSIFICATION: dict[str, type[BaseModel]] = {}


def register(task: str, name: str):
    """Decorator: bir BaseModel alt sınıfını registry'e kaydet."""

    def decorator(cls: type[BaseModel]) -> type[BaseModel]:
        cls.name = name
        cls.task = task  # type: ignore[assignment]
        if task == "regression":
            MODELS_REGRESSION[name] = cls
        elif task in ("classification", "binary_classification"):
            MODELS_CLASSIFICATION[name] = cls
        else:
            raise ValueError(f"Bilinmeyen task: {task!r}")
        return cls

    return decorator


def _ensure_models_imported() -> None:
    """Tüm concrete model dosyalarını yükleyip registry'i doldur.

    Modüller import edildikleri anda ``@register`` decorator'ları çalışır.
    """
    from src.m06_models.regression import (  # noqa: F401
        ridge,
        random_forest,
        lightgbm as lgbm_reg,
        stacking,
        pls,
        huber,
    )
    from src.m06_models.classification import (  # noqa: F401
        random_forest as rf_cls,
        lightgbm as lgbm_cls,
        stacking as stacking_cls,
        pheur_binary,
    )
    try:
        from src.m06_models.deep_learning import (  # noqa: F401
            cnn1d,
            resnet1d,
            mlp,
            rnn,
            transformer,
            autoencoder,
            cnn_lstm,
        )
    except Exception as exc:
        log.warning("Deep learning modelleri yüklenemedi (PyTorch?): %s", exc)


# ---- Yüksek seviyeli pipeline orkestrasyonu --------------------------------

REGRESSION_TARGETS: dict[str, str] = {
    "chlorophyll": "y_chl",
    "flavonol": "y_flav",
    "nbi": "y_nbi",
}

CLASSIFICATION_TARGETS: dict[str, str] = {
    "stress": "y_stress",
}


STRESS_CLASS_NAMES = ["sağlıklı", "flavescence dorée", "diğer biyotik stres", "abiyotik / diğer"]
PHEUR_CLASS_NAMES = ["FAIL (<eşik)", "PASS (>=eşik)"]


def _build_model(model_cls: type[BaseModel], cfg) -> BaseModel:
    """Config'ten ortak hiperparametreleri (cv, seed) ile model örneği üret."""
    # base hp from task-specific section (regression/classification)
    base_hp = cfg.get(f"models.{_task_for_cls(model_cls)}.{model_cls.name}", {}) or {}

    # deep-learning models read their hp from models.deep_learning.<model>
    hp = dict(base_hp)
    if getattr(model_cls, "is_deep_learning", False):
        dl_common = cfg.get("models.deep_learning.common", {}) or {}
        dl_model = cfg.get(f"models.deep_learning.{model_cls.name}", {}) or {}
        # merge: base_hp < common < model-specific
        merged = {**dl_common, **dl_model}
        hp = {**hp, **merged}

    return model_cls(
        config={
            "cv": cfg.get("models.cv", 5),
            "random_state": cfg.get("models.random_state", 42),
            "hp": hp,
            "log_transform_targets": cfg.get("models.log_transform_targets", []) or [],
            # GÖREV 1: SMOTE+Tomek için classification.resampling.* aktarımı
            "resampling_enabled": bool(cfg.get("classification.resampling.enabled", False)),
            "resampling_method": str(cfg.get("classification.resampling.method", "none")),
        }
    )


def _task_for_cls(cls: type[BaseModel]) -> str:
    """Modelin görev adını config sözlüğündeki anahtara dönüştür."""
    if cls.task == "regression":
        return "regression"
    return "classification"


def _select_models(task: str, cfg) -> dict[str, type[BaseModel]]:
    if task == "regression":
        return MODELS_REGRESSION
    if task == "classification":
        return MODELS_CLASSIFICATION
    if task == "deep_learning":
        # DL modelleri hem regression hem classification olarak çalıştırılır
        return {**MODELS_REGRESSION, **MODELS_CLASSIFICATION}
    raise ValueError(f"Bilinmeyen task: {task!r}")


def _is_deep(model_cls: type[BaseModel]) -> bool:
    return getattr(model_cls, "is_deep_learning", False)


def run_all(cfg, task: str) -> dict[str, Any]:
    """Bir pipeline aşamasını çalıştır: tüm modeller × tüm hedefler.

    Parameters
    ----------
    cfg
        ``Config`` nesnesi (config/default.yaml'den).
    task
        ``"regression"`` | ``"classification"`` | ``"deep_learning"``.
    """
    t0 = time.time()
    _ensure_models_imported()

    # GÖREV 1: classification + resampling enabled iken çıktılar
    # ``07_classification_resampled`` altına yazılır; baseline korunur.
    resampling_on = (
        task == "classification"
        and bool(cfg.get("classification.resampling.enabled", False))
        and str(cfg.get("classification.resampling.method", "none")) != "none"
    )
    stage = {
        "regression": "06_regression",
        "classification": "07_classification_resampled" if resampling_on else "07_classification",
        "deep_learning": "08_deep_learning",
    }[task]
    stage_dir = paths.stage_dir(stage)
    paths.write_source_marker(stage_dir, producer=f"src/m06_models/registry.py · {task}",
                              config_source=cfg.source)

    data = dataset_builder.load()
    X = data["X"]
    feature_names = data["feature_names"]

    results: dict[str, Any] = {}
    models_dict = _select_models(task, cfg)

    if task == "regression" or task == "deep_learning":
        for target_name, key in REGRESSION_TARGETS.items():
            y = data[key]
            for model_name, model_cls in models_dict.items():
                if model_cls.task != "regression":
                    continue
                if task == "deep_learning" and not _is_deep(model_cls):
                    continue
                if task == "regression" and _is_deep(model_cls):
                    continue
                try:
                    out_dir = stage_dir / model_name / target_name
                    model = _build_model(model_cls, cfg)
                    res = model.run(X, y, target_name, out_dir,
                                    feature_names=feature_names)
                    results[f"{model_name}/{target_name}"] = res["metrics"]
                except Exception as exc:
                    log.exception("HATA %s/%s: %s", model_name, target_name, exc)

    if task == "classification" or task == "deep_learning":
        for target_name, key in CLASSIFICATION_TARGETS.items():
            y = data[key]
            for model_name, model_cls in models_dict.items():
                if model_cls.task != "classification":
                    continue
                if task == "deep_learning" and not _is_deep(model_cls):
                    continue
                if task == "classification" and _is_deep(model_cls):
                    continue
                try:
                    out_dir = stage_dir / model_name / target_name
                    model = _build_model(model_cls, cfg)
                    res = model.run(X, y, target_name, out_dir,
                                    feature_names=feature_names,
                                    class_names=STRESS_CLASS_NAMES)
                    results[f"{model_name}/{target_name}"] = res["metrics"]
                except Exception as exc:
                    log.exception("HATA %s/%s: %s", model_name, target_name, exc)

        # Ph.Eur. binary — sadece klasik aşamada
        if task == "classification" and "pheur" in models_dict:
            try:
                threshold = float(cfg.get("targets.pheur_flavonol_threshold", 3.5))
                y_bin = pheur_pass_fail(data["y_flav"], threshold=threshold)
                model_cls = models_dict["pheur"]
                out_dir = stage_dir / "pheur_binary" / "flavonol_pheur"
                model = _build_model(model_cls, cfg)
                res = model.run(X, y_bin, "flavonol_pheur", out_dir,
                                feature_names=feature_names,
                                class_names=PHEUR_CLASS_NAMES)
                results["pheur_binary/flavonol_pheur"] = res["metrics"]
            except Exception as exc:
                log.exception("HATA pheur_binary: %s", exc)

    log.info("%s tamamlandı: %d run, süre=%.1fs", task, len(results), time.time() - t0)
    return results
