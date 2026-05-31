import numpy as np
from src.core import config
from src.m06_models import registry as reg


def _shorten_dl_hp(cfg):
    # ensure small training loops for CI
    data = cfg._data
    models = data.setdefault("models", {})
    dl = models.setdefault("deep_learning", {})
    dl_common = dl.setdefault("common", {})
    dl_common.update({
        "epochs": 1,
        "pretrain_epochs": 1,
        "finetune_epochs": 1,
        "batch_size": 16,
        "patience": 1,
    })


def test_models_run_quick(tmp_path):
    cfg = config.load("config/default.yaml")
    _shorten_dl_hp(cfg)
    reg._ensure_models_imported()

    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 12))
    y_reg = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=X.shape[0])
    y_clf = (rng.rand(X.shape[0]) > 0.5).astype(int)

    # test regression models
    for name, cls in reg.MODELS_REGRESSION.items():
        model = reg._build_model(cls, cfg)
        # speed up deep models further
        hp = model.config.get("hp", {}) or {}
        hp.setdefault("epochs", 1)
        hp.setdefault("batch_size", 8)
        model.config["hp"] = hp
        out = tmp_path / "models" / "regression" / name
        res = model.run(X, y_reg, "test", out)
        assert "metrics" in res

    # test classification models
    for name, cls in reg.MODELS_CLASSIFICATION.items():
        model = reg._build_model(cls, cfg)
        hp = model.config.get("hp", {}) or {}
        hp.setdefault("epochs", 1)
        hp.setdefault("batch_size", 8)
        model.config["hp"] = hp
        out = tmp_path / "models" / "classification" / name
        res = model.run(X, y_clf, "test", out)
        assert "metrics" in res
import numpy as np

from src.core import config
from src.m06_models import registry as reg


def test_registry_models_fit_predict():
    cfg = config.load("config/default.yaml")
    reg._ensure_models_imported()

    X = np.random.RandomState(0).randn(20, 52)

    # Regression models
    for name, cls in reg.MODELS_REGRESSION.items():
        model = reg._build_model(cls, cfg)
        y = np.random.RandomState(1).randn(20)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert len(y_pred) == 20

    # Classification models
    for name, cls in reg.MODELS_CLASSIFICATION.items():
        model = reg._build_model(cls, cfg)
        y = np.random.randint(0, 4, size=20)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert len(y_pred) == 20
