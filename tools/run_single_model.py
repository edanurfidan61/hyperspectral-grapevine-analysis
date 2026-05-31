#!/usr/bin/env python3
"""Run a single model locally for debugging (uses registry helpers)."""

from __future__ import annotations

import argparse

from src.core import config, logging_setup, paths
from src.m05_dataset import builder as dataset_builder
from src.m06_models import registry as reg


def main_cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--model", required=True)
    p.add_argument("--task", choices=["regression", "classification"], default="regression")
    p.add_argument("--target", required=True, help="e.g. flavonol, chlorophyll, nbi, stress")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cfg = config.load(args.config)
    logging_setup.init(cfg)
    paths.ensure_outputs_tree()

    data = dataset_builder.load()
    X = data["X"]

    reg._ensure_models_imported()
    if args.task == "regression":
        model_cls = reg.MODELS_REGRESSION.get(args.model)
        target_map = reg.REGRESSION_TARGETS
    else:
        model_cls = reg.MODELS_CLASSIFICATION.get(args.model)
        target_map = reg.CLASSIFICATION_TARGETS

    if model_cls is None:
        raise ValueError(f"Model bulunamadı: {args.model}")
    if args.target not in target_map:
        raise ValueError(f"Bilinmeyen target: {args.target}")

    y = data[target_map[args.target]]
    model = reg._build_model(model_cls, cfg)
    out_dir = paths.stage_dir("debug") / args.model / args.target

    # Fit + run (will save metrics in out_dir)
    model.fit(X, y)
    res = model.run(X, y, args.target, out_dir, feature_names=data.get("feature_names"))
    print("Tamam. Metrikler:", res.get("metrics"))


if __name__ == "__main__":
    main_cli()
