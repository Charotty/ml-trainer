"""Block 3: nested baselines, RF-dominance, sample weighting, feature ablation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ensemble import AdaptiveEnsembleTrainer  # noqa: E402
from src.features.phase1_schema import BASE_FEATURES, TARGET_NAMES  # noqa: E402
from src.models.ablation import (  # noqa: E402
    PRODUCTION_DROP_FORBIDDEN_NOTE,
    evaluate_ablation_nested,
    group_drop_spec,
)
from src.models.artifact_manifest import apply_oof_to_manifest, oof_placeholder  # noqa: E402
from src.models.baselines import (  # noqa: E402
    BASELINE_KINDS,
    evaluate_baselines_nested,
    simplicity_rule,
)


def _synthetic(
    n_patients: int = 8,
    *,
    seed: int = 0,
    with_demographics: bool = True,
    with_na_groups: bool = False,
    all_targets: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    targets = list(TARGET_NAMES if all_targets else TARGET_NAMES[:2])
    rows = []
    for i in range(n_patients):
        pid = f"P{i:02d}"
        row = {col: float(100.0 + rng.normal(0, 2)) for col in BASE_FEATURES}
        row["full_name"] = pid
        for t in targets:
            row[t] = float(rng.normal(0.0, 2.0) + (4.0 if t.endswith("_z") else 0.0) * (i % 3))
        if with_demographics:
            row["sex"] = 1.0 if i % 2 == 0 else 2.0
            row["age"] = 30.0 + i * 5
            row["bmi"] = 22.0 + (i % 4)
            row["body_type"] = float(i % 3)
            row["has_previous_surgery"] = float(i % 2)
            if i == 0:
                row["sex"] = np.nan
                row["age"] = np.nan
        if with_na_groups:
            for side in ("left", "right"):
                for axis in ("x", "y", "z"):
                    row[f"na_pop_shift_{side}_{axis}"] = 1.5
            row["na_sup_z_body_width_mm"] = 0.1 * i
            row["na_sup_pct_body_width_mm"] = 0.2 * i
        rows.append(row)
    return pd.DataFrame(rows)


def test_all_required_baselines_use_nested_helper() -> None:
    df = _synthetic(n_patients=8, with_demographics=True)
    report = evaluate_baselines_nested(
        df,
        kinds=BASELINE_KINDS,
        n_splits=2,
        trainer_kwargs={"estimator_profile": "tiny", "inner_n_splits": 2, "enrichment_mode": "none"},
        targets=[c for c in TARGET_NAMES[:2] if c in df.columns],
        yz_target_boost=False,
    )
    for kind in BASELINE_KINDS:
        assert kind in report["results"]
        payload = report["results"][kind]
        assert payload["protocol"] == "nested_groupkfold"
        assert "avg_mae_mm" in payload
        assert np.isfinite(payload["avg_mae_mm"])
    assert report["yz_target_boost"] is False
    assert report["simplicity_rule"]["selected"] in {"rf", "ensemble"}
    assert "inner_weight_summary" in report
    traces = report["inner_weight_summary"]["per_outer_fold_weights"]
    assert traces, "ensemble inner-fold weights must be persisted"


def test_simplicity_rule_prefers_rf_when_gain_inside_uncertainty() -> None:
    comparison = {
        "mean_delta_mm": 0.05,
        "delta_ci95": [-0.2, 0.3],
        "n_folds": 2,
    }
    decision = simplicity_rule(comparison, min_absolute_gain_mm=0.25)
    assert decision["selected"] == "rf"
    assert "uncertainty" in decision["reason"] or "threshold" in decision["reason"]


def test_yz_boost_optional_and_cv_uses_sample_weight() -> None:
    y = np.array([1.0, 2.0, 20.0, 21.0])
    trainer_boost = AdaptiveEnsembleTrainer(estimator_profile="tiny", yz_target_boost=True)
    trainer_plain = AdaptiveEnsembleTrainer(estimator_profile="tiny", yz_target_boost=False)
    boosted = trainer_boost._per_target_sample_weights("kidney_left_delta_z", y, None)
    plain = trainer_plain._per_target_sample_weights("kidney_left_delta_z", y, None)
    assert boosted is not None
    assert boosted[2] > boosted[0]
    assert plain is None

    row_w = np.array([1.0, 1.0, 4.0, 4.0])
    plain_row = trainer_plain._per_target_sample_weights("kidney_left_delta_x", y, row_w)
    np.testing.assert_allclose(plain_row, row_w)

    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=4, random_state=42, max_depth=2)
    X = np.arange(16, dtype=float).reshape(4, 4)
    y_fit = np.array([0.0, 0.0, 10.0, 10.0])
    w = np.array([0.01, 0.01, 5.0, 5.0])
    trainer_plain.cv_splitter = __import__("sklearn.model_selection", fromlist=["KFold"]).KFold(
        n_splits=2, shuffle=False
    )
    mae_w, _ = trainer_plain.evaluate_model_cv(model, X, y_fit, "RandomForest", sample_weight=w)
    mae_u, _ = trainer_plain.evaluate_model_cv(model, X, y_fit, "RandomForest", sample_weight=None)
    assert np.isfinite(mae_w) and np.isfinite(mae_u)


def test_ridge_one_hot_not_linear_sex_order() -> None:
    df = _synthetic(n_patients=8, with_demographics=True)
    trainer = AdaptiveEnsembleTrainer(
        estimator_profile="tiny",
        model_kind="ridge",
        encode_categoricals=True,
        enrichment_mode="none",
        yz_target_boost=False,
    )
    a = df.iloc[:4].reset_index(drop=True)
    b = df.iloc[4:].reset_index(drop=True)
    X_tr, X_te, y_tr, y_te = trainer.prepare_training_data_split(a, b)
    assert X_tr is not None
    names = trainer.feature_names
    assert "sex" not in names
    assert "body_type" not in names
    assert "sex_code_1" in names and "sex_code_2" in names
    assert "body_type_code_0" in names
    assert "sex_missing" in names
    assert "age_missing" in names


def test_ablation_api_does_not_authorize_production_drop() -> None:
    df = _synthetic(n_patients=8, with_na_groups=True, with_demographics=True)
    out = evaluate_ablation_nested(
        df,
        n_splits=2,
        model_kind="rf",
        groups=("demographics", "engineered", "na_pop_shift", "na_sup_affine"),
        targets=[c for c in TARGET_NAMES[:2] if c in df.columns],
    )
    assert out["allow_production_drop"] is False
    assert PRODUCTION_DROP_FORBIDDEN_NOTE in out["note"]
    assert "drop_demographics" in out["runs"]
    assert "drop_na_pop_shift" in out["runs"]
    assert "drop_na_sup_affine" in out["runs"]
    for key, delta in out["delta_vs_full"].items():
        assert delta["allow_production_drop"] is False
    spec = group_drop_spec("na_pop_shift")
    assert spec["drop_feature_prefixes"] == ("na_pop_shift_",)


def test_manifest_hooks_clinical_and_conformal_placeholders() -> None:
    empty = oof_placeholder(protocol="nested_groupkfold")
    assert empty["clinical_metrics"] is None
    assert empty["conformal"] is None
    assert empty["tail_metrics"] is None
    card = apply_oof_to_manifest(
        {"candidate_id": "x"},
        {
            "folds": [{"fold": 0}],
            "aggregated_metrics": {"avg_mae_mm": 1.0},
            "oof_protocol": "nested_groupkfold",
            "oof_status": "computed",
            "oof_weight_mode": "inner_groupkfold",
            "clinical_metrics": {"has_full_xyz": False},
            "subgroup_metrics": {"groups": {}, "no_difference_claims": False},
            "conformal": {"method": "oof_residual_conformal"},
            "tail_metrics": {"primary": "endpoint_error_3d_mm"},
            "worst_cases": [],
        },
    )
    assert card["clinical_metrics"]["has_full_xyz"] is False
    assert card["conformal"]["method"] == "oof_residual_conformal"
    assert card["subgroup_metrics"]["no_difference_claims"] is False
