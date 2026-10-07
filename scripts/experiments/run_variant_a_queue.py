#!/usr/bin/env python3
"""Sequential Variant A experiment queue (ordered journal appends)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COMPACT_FEATURES = [
    "bmi",
    "sex",
    "age",
    "body_type",
    "has_previous_surgery",
    "body_depth_mm",
    "body_width_mm",
    "body_area_mm2",
    "abd_wall_thickness_mm",
    "lumbar_lordosis_deg",
    "s1_plate_tilt_deg",
    "kidney_left_volume_cm3",
    "kidney_right_volume_cm3",
    "kidney_left_center_x_rel",
    "kidney_left_center_y_rel",
    "kidney_left_center_z_rel",
    "kidney_right_center_x_rel",
    "kidney_right_center_y_rel",
    "kidney_right_center_z_rel",
    "kidney_lr_sep_x",
    "kidney_lr_sep_y",
    "kidney_lr_sep_z",
    "kidney_left_z_span_supine_mm",
    "kidney_right_z_span_supine_mm",
]


def run(args: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "experiments" / "run_variant_a.py"), *args]
    print("RUN:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"FAILED exit={result.returncode} tag={args}", flush=True)


def main() -> int:
    # Stage 1 is already on disk (models/experiments/1_*.json) and journalled.
    # Later rows must beat that best (random forest). Enrichment matches the
    # production recipe (na_trends). Skip the xlsx rebuild.
    common = [
        "--skip-build",
        "--estimator-profile", "production",
        "--enrichment-mode", "na_trends",
        "--seeds", "0,1,2",
    ]
    # Tree ablations use the stage-1 winner. A full ensemble grid is several
    # hours per loss and would repeat four models when one already leads.
    tree = ["--model-kind", "rf"]

    # Stage 0: current production ensemble, holdout excluded.
    run([
        "--stage", "0", "--tag", "ensemble_current",
        "--change", "текущий ансамбль (na_trends, yz boost, production)",
        "--model-kind", "ensemble", *common,
    ])

    # Stage 2 yz boost modes on the winning tree model
    for mode, tag, change in [
        ("off", "yz_off", "лес: yz_target_boost выключен"),
        ("soft", "yz_soft", "лес: yz_target_boost мягкий (делитель 30, потолок 1.0)"),
        ("y_only", "yz_y_only", "лес: yz_target_boost только для Y"),
    ]:
        run([
            "--stage", "2", "--tag", tag, "--change", change,
            "--yz-boost-mode", mode, *tree, *common,
        ])

    # Stage 3 loss profiles
    for loss, tag, change in [
        ("absolute_error", "loss_ae", "loss_profile=absolute_error (RF+GBT)"),
        ("quantile_0.5", "loss_q50", "loss_profile=quantile alpha=0.5"),
        ("huber_0.5", "loss_h05", "loss_profile=huber alpha=0.5"),
        ("huber_0.7", "loss_h07", "loss_profile=huber alpha=0.7"),
        ("huber_0.9", "loss_h09", "loss_profile=huber alpha=0.9"),
        ("huber_all_axes", "loss_h_all", "huber на всех осях"),
    ]:
        kind = "gbt" if loss != "absolute_error" else "rf"
        run([
            "--stage", "3", "--tag", tag, "--change", change,
            "--model-kind", kind, "--loss-profile", loss, *common,
        ])

    # Stage 4 small_n + a couple of overrides
    run([
        "--stage", "4", "--tag", "small_n", "--change", "профиль small_n",
        "--model-kind", "rf", "--estimator-profile", "small_n",
        "--skip-build", "--enrichment-mode", "na_trends", "--seeds", "0,1,2",
    ])
    overrides = {
        "GradientBoosting": {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.03, "min_samples_leaf": 8, "subsample": 0.7},
        "RandomForest": {"max_depth": 4, "min_samples_leaf": 15, "max_features": 0.3},
    }
    run([
        "--stage", "4", "--tag", "small_n_grid_a",
        "--change", "small_n + GBT depth3/300/lr0.03 + RF depth4",
        "--model-kind", "gbt", "--estimator-profile", "small_n",
        "--estimator-overrides", json.dumps(overrides),
        "--skip-build", "--enrichment-mode", "na_trends", "--seeds", "0,1,2",
    ])

    # Stage 5 feature ablations
    for groups, tag, change in [
        (["engineered"], "drop_engineered_naish", "без синтетических/engineered групп"),
    ]:
        run([
            "--stage", "5", "--tag", tag, "--change", change,
            "--model-kind", "rf", "--drop-feature-groups", *groups, *common,
        ])
    run([
        "--stage", "5", "--tag", "drop_na_trends_prefixes",
        "--change", "без na_trends префиксов",
        "--model-kind", "rf",
        "--drop-feature-prefixes", "na_pop_shift_", "na_sup_z_", "na_sup_pct_", "kits_z_", "kits_pct_",
        *common,
    ])
    run([
        "--stage", "5", "--tag", "drop_na_pop_shift",
        "--change", "без na_pop_shift_",
        "--model-kind", "rf",
        "--drop-feature-prefixes", "na_pop_shift_",
        *common,
    ])
    run([
        "--stage", "5", "--tag", "no_missing_indicators",
        "--change", "без индикаторов пропусков",
        "--model-kind", "rf", "--no-add-missing-indicators", *common,
    ])
    run([
        "--stage", "5", "--tag", "compact25",
        "--change", "компактный набор ~24 признаков",
        "--model-kind", "rf",
        "--keep-feature-names", *COMPACT_FEATURES,
        *common,
    ])
    run([
        "--stage", "5", "--tag", "drop_body_com",
        "--change", "без body_com_* и производных",
        "--model-kind", "rf",
        "--drop-feature-prefixes", "body_com_", "body_com_to_", "kidney_left_to_body", "kidney_right_to_body", "body_center_",
        *common,
    ])
    run([
        "--stage", "5", "--tag", "no_na_trends",
        "--change", "без когортных na_trends",
        "--model-kind", "rf",
        "--enrichment-mode", "none",
        "--skip-build", "--estimator-profile", "production", "--seeds", "0,1,2",
    ])
    run([
        "--stage", "5", "--tag", "per_axis",
        "--change", "отдельный набор признаков на каждую ось",
        "--model-kind", "rf", "--per-axis-features", *common,
    ])

    # Stage 6: weight schemes on the small_n ensemble (same question, much
    # less tree depth, so the comparison is about the weights).
    ens = [
        "--skip-build", "--estimator-profile", "small_n",
        "--enrichment-mode", "na_trends", "--seeds", "0,1,2",
        "--model-kind", "ensemble",
    ]
    for mode, tag, change in [
        ("equal", "ens_equal", "равные веса, профиль small_n"),
        ("fixed_prior", "ens_fixed_prior", "фиксированные стартовые веса, small_n"),
        ("shrink", "ens_shrink", "сжатие весов к равным, small_n"),
        ("stacking", "ens_stacking", "стекинг неотрицательным риджем, small_n"),
        ("best_per_axis", "ens_best_axis", "одна лучшая модель на ось, small_n"),
    ]:
        run([
            "--stage", "6", "--tag", tag, "--change", change,
            "--ensemble-weight-mode", mode, *ens,
        ])

    # Stage 7 plugin models (each kind is already its own module).
    for kind, tag, change in [
        ("elasticnet", "enet", "ElasticNet, внутренний подбор alpha"),
        ("huber_linear", "huber", "HuberRegressor"),
        ("pls", "pls", "PLS по одной оси"),
        ("pls_multi", "pls_multi", "PLS сразу по трём осям почки"),
        ("gpr", "gpr", "гауссовский процесс RBF"),
        ("svr_linear", "svr_lin", "SVR линейное ядро"),
        ("svr_rbf", "svr_rbf", "SVR ядро RBF"),
        ("knn", "knn", "k ближайших соседей"),
        ("lightgbm", "lgbm", "LightGBM, неглубокие деревья"),
        ("mtenet", "mtenet", "MultiTaskElasticNet по осям почки"),
        ("chain", "chain", "цепочка регрессоров X, затем Y, затем Z"),
        ("joint_z", "joint_z", "совместная Z левой и правой почки"),
    ]:
        run([
            "--stage", "7", "--tag", tag, "--change", change,
            "--model-kind", kind, *common,
        ])

    # Stage 8 on the stage-1 winner, then again from whatever the journal accepted.
    for zpp, tag, change in [
        ("median_shrink", "z_median_shrink", "сжатие Z к медиане"),
        ("clip_q05_q95", "z_clip", "ограничение Z квантилями 5–95%"),
        ("sign_magnitude", "z_sign_mag", "знак Z отдельно, величина сжата к медиане"),
    ]:
        run([
            "--stage", "8", "--tag", tag, "--change", change,
            "--model-kind", "rf", "--z-postprocess", zpp, *common,
        ])
    run([
        "--stage", "8", "--tag", "z_calibrator",
        "--change", "калибратор Z по отложенным прогнозам",
        "--model-kind", "rf", "--z-calibrator", *common,
    ])
    run([
        "--stage", "8", "--tag", "z_on_best",
        "--change", "сжатие Z к медиане поверх лучшей конфигурации",
        "--base-from-best", "--z-postprocess", "median_shrink",
        *common,
    ])
    return 0


def resume() -> int:
    """Continue after the 2026-10-01 stop (journal ends at huber 0.7).

    Every remaining run turns sample-weight boost off, because that is the
    accepted configuration (run 2_yz_off, MAE 7.31 mm).
    """
    common = [
        "--skip-build",
        "--estimator-profile", "production",
        "--enrichment-mode", "na_trends",
        "--seeds", "0,1,2",
        "--yz-boost-mode", "off",
    ]
    for loss, tag, change in [
        ("huber_0.9", "loss_h09", "бустинг, huber alpha=0.9, без усиления YZ"),
        ("huber_all_axes", "loss_h_all", "бустинг, huber на всех осях, без усиления YZ"),
    ]:
        run([
            "--stage", "3", "--tag", tag, "--change", change,
            "--model-kind", "gbt", "--loss-profile", loss, *common,
        ])

    run([
        "--stage", "4", "--tag", "small_n",
        "--change", "лес small_n, без усиления YZ",
        "--model-kind", "rf", "--estimator-profile", "small_n",
        "--skip-build", "--enrichment-mode", "na_trends", "--seeds", "0,1,2",
        "--yz-boost-mode", "off",
    ])
    overrides = {
        "GradientBoosting": {
            "max_depth": 3,
            "n_estimators": 300,
            "learning_rate": 0.03,
            "min_samples_leaf": 8,
            "subsample": 0.7,
        },
        "RandomForest": {"max_depth": 4, "min_samples_leaf": 15, "max_features": 0.3},
    }
    run([
        "--stage", "4", "--tag", "small_n_grid_a",
        "--change", "бустинг small_n depth3, без усиления YZ",
        "--model-kind", "gbt", "--estimator-profile", "small_n",
        "--estimator-overrides", json.dumps(overrides),
        "--skip-build", "--enrichment-mode", "na_trends", "--seeds", "0,1,2",
        "--yz-boost-mode", "off",
    ])

    run([
        "--stage", "5", "--tag", "drop_engineered",
        "--change", "лес без engineered-группы, без усиления YZ",
        "--model-kind", "rf", "--drop-feature-groups", "engineered", *common,
    ])
    run([
        "--stage", "5", "--tag", "drop_na_trends_prefixes",
        "--change", "лес без префиксов na_trends, без усиления YZ",
        "--model-kind", "rf",
        "--drop-feature-prefixes", "na_pop_shift_", "na_sup_z_", "na_sup_pct_", "kits_z_", "kits_pct_",
        *common,
    ])
    run([
        "--stage", "5", "--tag", "drop_na_pop_shift",
        "--change", "лес без na_pop_shift_, без усиления YZ",
        "--model-kind", "rf", "--drop-feature-prefixes", "na_pop_shift_", *common,
    ])
    run([
        "--stage", "5", "--tag", "no_missing_indicators",
        "--change", "лес без индикаторов пропусков, без усиления YZ",
        "--model-kind", "rf", "--no-add-missing-indicators", *common,
    ])
    run([
        "--stage", "5", "--tag", "compact25",
        "--change", "лес, компактный набор ~24 признаков, без усиления YZ",
        "--model-kind", "rf", "--keep-feature-names", *COMPACT_FEATURES, *common,
    ])
    run([
        "--stage", "5", "--tag", "drop_body_com",
        "--change", "лес без body_com и производных, без усиления YZ",
        "--model-kind", "rf",
        "--drop-feature-prefixes", "body_com_", "body_com_to_", "kidney_left_to_body", "kidney_right_to_body", "body_center_",
        *common,
    ])
    run([
        "--stage", "5", "--tag", "no_na_trends",
        "--change", "лес без когортных na_trends, без усиления YZ",
        "--model-kind", "rf", "--enrichment-mode", "none",
        "--skip-build", "--estimator-profile", "production", "--seeds", "0,1,2",
        "--yz-boost-mode", "off",
    ])
    run([
        "--stage", "5", "--tag", "per_axis",
        "--change", "лес, отдельный набор признаков на ось, без усиления YZ",
        "--model-kind", "rf", "--per-axis-features", *common,
    ])

    ens = [
        "--skip-build", "--estimator-profile", "small_n",
        "--enrichment-mode", "na_trends", "--seeds", "0,1,2",
        "--yz-boost-mode", "off", "--model-kind", "ensemble",
    ]
    for mode, tag, change in [
        ("equal", "ens_equal", "равные веса, small_n, без усиления YZ"),
        ("fixed_prior", "ens_fixed_prior", "фиксированные стартовые веса, small_n, без усиления YZ"),
        ("shrink", "ens_shrink", "сжатие весов к равным, small_n, без усиления YZ"),
        ("stacking", "ens_stacking", "стекинг, small_n, без усиления YZ"),
        ("best_per_axis", "ens_best_axis", "одна модель на ось, small_n, без усиления YZ"),
    ]:
        run([
            "--stage", "6", "--tag", tag, "--change", change,
            "--ensemble-weight-mode", mode, *ens,
        ])

    for kind, tag, change in [
        ("elasticnet", "enet", "ElasticNet, без усиления YZ"),
        ("huber_linear", "huber", "HuberRegressor, без усиления YZ"),
        ("pls", "pls", "PLS по одной оси, без усиления YZ"),
        ("pls_multi", "pls_multi", "PLS по трём осям почки, без усиления YZ"),
        ("gpr", "gpr", "гауссовский процесс RBF, без усиления YZ"),
        ("svr_linear", "svr_lin", "SVR линейное ядро, без усиления YZ"),
        ("svr_rbf", "svr_rbf", "SVR ядро RBF, без усиления YZ"),
        ("knn", "knn", "k ближайших соседей, без усиления YZ"),
        ("lightgbm", "lgbm", "LightGBM, без усиления YZ"),
        ("mtenet", "mtenet", "MultiTaskElasticNet, без усиления YZ"),
        ("chain", "chain", "цепочка X, затем Y, затем Z, без усиления YZ"),
        ("joint_z", "joint_z", "совместная Z левой и правой, без усиления YZ"),
    ]:
        run([
            "--stage", "7", "--tag", tag, "--change", change,
            "--model-kind", kind, *common,
        ])

    for zpp, tag, change in [
        ("median_shrink", "z_median_shrink", "сжатие Z к медиане, лес без усиления YZ"),
        ("clip_q05_q95", "z_clip", "ограничение Z квантилями 5–95%, лес без усиления YZ"),
        ("sign_magnitude", "z_sign_mag", "знак Z отдельно, величина сжата, лес без усиления YZ"),
    ]:
        run([
            "--stage", "8", "--tag", tag, "--change", change,
            "--model-kind", "rf", "--z-postprocess", zpp, *common,
        ])
    run([
        "--stage", "8", "--tag", "z_calibrator",
        "--change", "калибратор Z по отложенным прогнозам, лес без усиления YZ",
        "--model-kind", "rf", "--z-calibrator", *common,
    ])
    run([
        "--stage", "8", "--tag", "z_on_best",
        "--change", "сжатие Z к медиане поверх лучшей конфигурации",
        "--base-from-best", "--z-postprocess", "median_shrink", *common,
    ])
    return 0


if __name__ == "__main__":
    if "--resume" in sys.argv:
        raise SystemExit(resume())
    raise SystemExit(main())
