#!/usr/bin/env python3
"""Pipeline tek giriş noktası: `python main.py` ile tüm aşamalar çalıştırılır."""

from __future__ import annotations

import argparse
import time

from src.core import config, logging_setup, paths
from src.core.logging_setup import get as get_logger
from src.m04_features import shap_analysis as shap_mod
from src.m04_features import rfe_selection as rfe_mod
from src.m04_features import ga_feature_selection as ga_mod
from src.m04_features import ga_wavelength_viz as ga_viz_mod
from src.m04_features import pls_vip as pls_vip_mod
from src.m04_features import feature_consensus as consensus_mod
from src.m05_dataset import builder as ds
from src.m05_dataset import outlier_filter as outlier_mod
from src.m05_dataset import holdout_split as holdout_mod
from src.m06_models import eda as eda_module
from src.m05_dataset import visualize as viz
from src.m06_models import registry as reg
from src.m06_models import tuning as tuning_mod
from src.m06_models import ablation as ablation_mod
from src.m06_models import final_combos as final_combos_mod
from src.m06_models import ordinal_flavonol as ord_flav
from src.m06_models import anomaly_flavonol as anom_flav
from src.m06_models import flavonol_combos as flav_combos
from src.m07_ensemble import ensemble as ens
from src.m07_ensemble import model_summary as model_sum_mod

log = get_logger("main")


def build_pipeline(cfg, force: bool = False) -> list[tuple[str, object]]:
    """Pipeline aşamalarını (ad, çağrılabilir) listesi olarak üret.

    Tek kaynak: hem ``main()`` hem de ``run_all.py`` bu listeyi kullanır; aşama
    sırası burada değişir, iki giriş noktası da otomatik aynı sırayı görür.
    """
    return [
        ("01_dataset", lambda: ds.build(cfg, force=force)),
        ("01b_outliers", lambda: outlier_mod.run(cfg, force=force)),
        ("01c_holdout", lambda: holdout_mod.run(cfg, force=force)),
        ("02_eda", lambda: eda_module.run(cfg)),
        ("03_visualization", lambda: viz.run(cfg, force=force)),
        ("04_feature_shap", lambda: shap_mod.run(cfg)),
        ("05_feature_rfe", lambda: rfe_mod.run(cfg)),
        ("05b_pls_vip", lambda: pls_vip_mod.run(cfg)),
        ("06_regression", lambda: reg.run_all(cfg, task="regression")),
        ("07_classification", lambda: reg.run_all(cfg, task="classification")),
        ("06b_regression_tuned", lambda: tuning_mod.run(cfg, task="regression")),
        ("07b_classification_tuned", lambda: tuning_mod.run(cfg, task="classification")),
        ("08_deep_learning", lambda: reg.run_all(cfg, task="deep_learning")),
        ("09_ordinal_flavonol", lambda: ord_flav.run(cfg)),
        ("10_anomaly_flavonol", lambda: anom_flav.run(cfg)),
        ("11_ensemble", lambda: ens.run(cfg)),
        ("12_ga_feature_selection", lambda: ga_mod.run(cfg)),
        ("12b_ga_wavelength_consensus", lambda: ga_viz_mod.run(cfg)),
        ("13_flavonol_combos", lambda: flav_combos.run(cfg)),
        ("13b_feature_consensus", lambda: consensus_mod.run(cfg)),
        ("14_model_summary", lambda: model_sum_mod.run(cfg)),
        ("15_ablation", lambda: ablation_mod.run(cfg)),
        ("16_final_combos", lambda: final_combos_mod.run(cfg)),
    ]


def main(cfg_path: str = "config/default.yaml", stages: list[str] | str = "all", force: bool = False):
    cfg = config.load(cfg_path)
    logging_setup.init(cfg)
    log = logging_setup.get("main")
    paths.ensure_outputs_tree()
    manifest = paths.start_manifest(cfg)

    pipeline = build_pipeline(cfg, force=force)

    for name, fn in pipeline:
        selectable = stages != "all" and name not in (stages if isinstance(stages, (list, tuple)) else [stages])
        if selectable:
            continue
        log.info("=== %s başlıyor ===", name)
        t0 = time.time()
        try:
            fn()
            manifest.record_stage(name, duration=time.time() - t0, status="ok")
        except Exception as exc:
            log.exception("Aşama %s hata: %s", name, exc)
            manifest.record_stage(name, duration=time.time() - t0, status="failed")
        log.info("=== %s tamamlandı (%.1fs) ===", name, time.time() - t0)

    manifest.write()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--stages", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    stages_arg = "all" if not args.stages else args.stages
    main(args.config, stages_arg, args.force)
