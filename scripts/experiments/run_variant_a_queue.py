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
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    # Stages after 0 assume zero-row already recorded. Skip-build after first rebuild.
    common = ["--skip-build", "--estimator-profile", "production", "--enrichment-mode", "none", "--seeds", "0,1,2"]

    # Stage 1 baselines
    for kind, tag, change in [
        ("median", "median", "базова: глобальная медиана"),
        ("group_median", "group_median", "базис: медиана по полу и телосложению"),
        ("ridge", "ridge", "базис: гребневая регрессия"),
        ("rf", "rf", "базис: случайный лес"),
        ("gbt", "gbt", "базис: градиентный бустинг"),
    ]:
        run(["--stage", "1", "--tag", tag, "--change", change, "--model-kind", kind, *common])

    # Stage 2 yz boost modes
    for mode, tag, change in [
        ("off", "yz_off", "yz_target_boost выключен"),
        ("soft", "yz_soft", "yz_target_boost мягкий (делитель 30, потолок 1.0)"),
        ("y_only", "yz_y_only", "yz_target_boost только для Y"),
    ]:
        run([
            "--stage", "2", "--tag", tag, "--change", change,
            "--model-kind", "ensemble", "--yz-boost-mode", mode, *common,
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
        run([
            "--stage", "3", "--tag", tag, "--change", change,
            "--model-kind", "ensemble", "--loss-profile", loss, *common,
        ])

    # Stage 4 small_n + a couple of overrides
    run([
        "--stage", "4", "--tag", "small_n", "--change", "профиль small_n",
        "--model-kind", "ensemble", "--estimator-profile", "small_n",
        "--skip-build", "--enrichment-mode", "none", "--seeds", "0,1,2",
    ])
    overrides = {
        "GradientBoosting": {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.03, "min_samples_leaf": 8, "subsample": 0.7},
        "RandomForest": {"max_depth": 4, "min_samples_leaf": 15, "max_features": 0.3},
    }
    run([
        "--stage", "4", "--tag", "small_n_grid_a",
        "--change", "small_n + GBT depth3/300/lr0.03 + RF depth4",
        "--model-kind", "ensemble", "--estimator-profile", "small_n",
        "--estimator-overrides", json.dumps(overrides),
        "--skip-build", "--enrichment-mode", "none", "--seeds", "0,1,2",
    ])

    # Stage 5 feature ablations
    for groups, tag, change in [
        (["engineered"], "drop_engineered_naish", "без синтетических/engineered групп"),
    ]:
        run([
            "--stage", "5", "--tag", tag, "--change", change,
            "--model-kind", "ensemble", "--drop-feature-groups", *groups, *common,
        ])
    run([
        "--stage", "5", "--tag", "drop_na_trends_prefixes",
        "--change", "без na_trends префиксов",
        "--model-kind", "ensemble",
        "--drop-feature-prefixes", "na_pop_shift_", "na_sup_z_", "na_sup_pct_", "kits_z_", "kits_pct_",
        *common,
    ])
    run([
        "--stage", "5", "--tag", "drop_na_pop_shift",
        "--change", "без na_pop_shift_",
        "--model-kind", "ensemble",
        "--drop-feature-prefixes", "na_pop_shift_",
        *common,
    ])
    run([
        "--stage", "5", "--tag", "no_missing_indicators",
        "--change", "без индикаторов пропусков",
        "--model-kind", "ensemble", "--no-add-missing-indicators", *common,
    ])
    run([
        "--stage", "5", "--tag", "compact25",
        "--change", "компактный набор ~24 признаков",
        "--model-kind", "ensemble",
        "--keep-feature-names", *COMPACT_FEATURES,
        *common,
    ])
    run([
        "--stage", "5", "--tag", "drop_body_com",
        "--change", "без body_com_* и производных",
        "--model-kind", "ensemble",
        "--drop-feature-prefixes", "body_com_", "body_com_to_", "kidney_left_to_body", "kidney_right_to_body", "body_center_",
        *common,
    ])

    # Stage 6 ensemble weight modes
    for mode, tag, change in [
        ("equal", "ens_equal", "равные веса ансамбля"),
        ("fixed_prior", "ens_fixed_prior", "фиксированные стартовые веса"),
        ("shrink", "ens_shrink", "сжатие весов к равным"),
        ("stacking", "ens_stacking", "стек/сжатие к равным (proxy)"),
        ("best_per_axis", "ens_best_axis", "одна лучшая модель на ось"),
    ]:
        run([
            "--stage", "6", "--tag", tag, "--change", change,
            "--model-kind", "ensemble", "--ensemble-weight-mode", mode, *common,
        ])

    # Stage 8 Z postprocess (after models land; uses current ensemble)
    for zpp, tag, change in [
        ("median_shrink", "z_median_shrink", "сжатие Z к медиане"),
        ("clip_q05_q95", "z_clip", "ограничение Z квантилями 5-95%"),
    ]:
        run([
            "--stage", "8", "--tag", tag, "--change", change,
            "--model-kind", "ensemble", "--z-postprocess", zpp, *common,
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
