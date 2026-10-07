"""Block 4: 3D/tail metrics, conformal uncertainty, missingness, subgroups."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.kidney_displacement_api import prediction_uncertainty_payload  # noqa: E402
from src.data.excel_displacement_adapter import _sex_to_vybor_code  # noqa: E402
from src.features.fold_categoricals import (  # noqa: E402
    FoldCategoricalEncoder,
    bmi_from_weight_height,
    encode_sex_label,
    parse_dicom_age,
)
from src.features.phase1_schema import BASE_FEATURES, TARGET_NAMES  # noqa: E402
from src.models.baselines import make_trainer_factory  # noqa: E402
from src.models.clinical_metrics import (  # noqa: E402
    compute_clinical_within_ratios,
    endpoint_errors_3d,
    subgroup_error_table,
)
from src.models.nested_cv import evaluate_nested_groupkfold_oof  # noqa: E402
from src.models.uncertainty import (  # noqa: E402
    METHOD_CONFORMAL,
    METHOD_UNAVAILABLE,
    UNAVAILABLE_NOTE,
    fit_conformal_from_oof,
    intervals_for_point_predictions,
)
import src.api.kidney_displacement_api as api_mod  # noqa: E402


def test_endpoint_error_is_euclidean_not_magnitude_diff() -> None:
    # Left true (3,4,0) vs pred (0,0,0) → 5 mm. Right exact match → 0.
    cols = list(TARGET_NAMES)
    y_true = pd.DataFrame(
        [[3.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
        columns=cols,
    )
    y_pred = pd.DataFrame(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        columns=cols,
    )
    err = endpoint_errors_3d(y_true, y_pred, cols)
    np.testing.assert_allclose(err["endpoint_error_left_mm"], [5.0])
    np.testing.assert_allclose(err["endpoint_error_right_mm"], [0.0])
    np.testing.assert_allclose(err["endpoint_error_mean_mm"], [2.5])

    summary, per = compute_clinical_within_ratios(y_true, y_pred, cols)
    assert summary["within_5mm_ratio"] == 1.0
    assert summary["within_15mm_ratio"] == 1.0
    assert "p90_endpoint_error_mm" in summary
    np.testing.assert_allclose(per["endpoint_error_left_mm"], [5.0])


def test_within_ratios_use_3d_endpoint_mean() -> None:
    cols = list(TARGET_NAMES)
    y_true = np.zeros((3, 6))
    y_pred = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # mean 0
            [6.0, 0.0, 0.0, 6.0, 0.0, 0.0],  # mean 6
            [16.0, 0.0, 0.0, 16.0, 0.0, 0.0],  # mean 16
        ]
    )
    summary, per = compute_clinical_within_ratios(y_true, y_pred, cols)
    np.testing.assert_allclose(per["endpoint_error_mean_mm"], [0.0, 6.0, 16.0])
    assert summary["within_5mm_ratio"] == pytest.approx(1.0 / 3.0)
    assert summary["within_10mm_ratio"] == pytest.approx(2.0 / 3.0)
    assert summary["within_15mm_ratio"] == pytest.approx(2.0 / 3.0)
    assert summary["max_endpoint_error_mm"] == pytest.approx(16.0)


def test_conformal_intervals_and_no_fake_confidence() -> None:
    rng = np.random.default_rng(0)
    n = 20
    truth = pd.DataFrame({"kidney_left_delta_x": rng.normal(0, 1, n)})
    pred = truth + rng.normal(0, 0.2, size=truth.shape)
    mask_a = np.zeros(n, dtype=bool)
    mask_a[:10] = True
    bundle = fit_conformal_from_oof(
        truth,
        pred,
        targets=["kidney_left_delta_x"],
        coverage=0.9,
        fold_masks=[mask_a, ~mask_a],
    )
    assert bundle["method"] == METHOD_CONFORMAL
    q = bundle["per_target"]["kidney_left_delta_x"]["q_hat_mm"]
    assert q > 0
    payload = intervals_for_point_predictions({"kidney_left_delta_x": 3.0}, bundle)
    assert payload["prediction_confidence"] is None
    lo = payload["intervals"]["kidney_left_delta_x"]["lower_mm"]
    hi = payload["intervals"]["kidney_left_delta_x"]["upper_mm"]
    assert lo < 3.0 < hi

    missing = intervals_for_point_predictions({"kidney_left_delta_x": 3.0}, None)
    assert missing["method"] == METHOD_UNAVAILABLE
    assert missing["prediction_confidence"] is None
    assert "1 - |pred|/50" in missing["note"] or "/50" in missing["note"]

    api_src = (ROOT / "src" / "api" / "kidney_displacement_api.py").read_text(encoding="utf-8")
    assert "1.0 - abs(pred) / 50.0" not in api_src
    assert "abs(pred)/50" not in api_src

    api_mod.model_data = None
    api_payload = prediction_uncertainty_payload({"kidney_left_delta_x": 12.0})
    assert api_payload["prediction_confidence"] is None
    assert api_payload["method"] == METHOD_UNAVAILABLE

    api_mod.model_data = {"conformal": bundle}
    api_ok = prediction_uncertainty_payload({"kidney_left_delta_x": 12.0})
    assert api_ok["method"] == METHOD_CONFORMAL
    assert api_ok["prediction_confidence"] is None
    api_mod.model_data = None


def test_nested_oof_includes_3d_conformal_and_subgroups() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for i in range(8):
        row = {col: float(100.0 + rng.normal(0, 2)) for col in BASE_FEATURES}
        row["full_name"] = f"P{i:02d}"
        for t in TARGET_NAMES:
            row[t] = float(rng.normal(0, 1.5))
        row["sex"] = 1.0 if i < 3 else 2.0
        row["age"] = 25 + 8 * i
        row["bmi"] = 20 + i
        row["body_type"] = float(i % 3)
        row["has_previous_surgery"] = float(i % 2)
        if i == 7:
            row["sex"] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    result = evaluate_nested_groupkfold_oof(
        df,
        trainer_factory=make_trainer_factory(
            "rf",
            estimator_profile="tiny",
            enrichment_mode="none",
            yz_target_boost=False,
        ),
        n_splits=2,
        targets=list(TARGET_NAMES),
    )
    assert result.clinical_metrics is not None
    assert result.clinical_metrics["has_full_xyz"] is True
    left = result.clinical_metrics["clinical_3d"]["left"]
    assert "p90_mm" in left and "within_15mm_ratio" in left
    assert result.conformal["method"] == METHOD_CONFORMAL
    fields = result.to_manifest_fields()
    assert fields["clinical_metrics"] is not None
    assert fields["conformal"]["method"] == METHOD_CONFORMAL
    subgroups = result.subgroup_metrics["groups"]
    assert "sex" in subgroups
    small = [row for rows in subgroups.values() for row in rows if row["n"] < 8]
    assert small
    assert all("do not claim" in row["inference_note"].lower() or row["n"] >= 8 for row in small)
    assert result.subgroup_metrics["no_difference_claims"] is False


def test_unknown_demographics_stay_missing() -> None:
    assert np.isnan(encode_sex_label(None))
    assert np.isnan(encode_sex_label("O"))
    assert np.isnan(encode_sex_label(0))
    assert encode_sex_label("M") == 1.0
    assert encode_sex_label("F") == 2.0
    assert np.isnan(_sex_to_vybor_code("unknown"))
    assert np.isnan(parse_dicom_age(None))
    assert parse_dicom_age("045Y") == 45.0
    assert np.isnan(bmi_from_weight_height(None, None))
    assert bmi_from_weight_height(80, 2.0) == pytest.approx(20.0)

    encoder = FoldCategoricalEncoder(add_missing_indicators=True)
    train = pd.DataFrame({"sex": [1.0, 2.0], "body_type": [0.0, 1.0], "age": [40.0, np.nan]})
    val = pd.DataFrame({"sex": [np.nan, 1.0], "body_type": [0.0, np.nan], "age": [np.nan, 50.0]})
    encoder.fit(train)
    out = encoder.transform(val)
    assert "sex" not in out.columns
    assert out.loc[0, "sex_code_1"] == 0.0
    assert out.loc[0, "sex_code_2"] == 0.0
    assert out.loc[0, "sex_missing"] == 1.0
    assert out.loc[0, "body_type_code_0"] == 1.0  # 0 is a real class
    assert out.loc[1, "body_type_missing"] == 1.0

    extractor = (ROOT / "scripts" / "inference" / "enhanced_ct_extractor.py").read_text(
        encoding="utf-8"
    )
    assert "age = 50.0" not in extractor
    assert "bmi = 25.0" not in extractor
    assert "sex = 0.0" not in extractor
    ds = SimpleNamespace(PatientSex=None, PatientAge=None, PatientWeight=None, PatientSize=None)
    sys.path.insert(0, str(ROOT / "scripts" / "inference"))
    from enhanced_ct_extractor import _extract_demographics  # noqa: E402

    demo = _extract_demographics(ds)
    assert demo["sex"] is None
    assert demo["age"] is None
    assert demo["bmi"] is None


def test_subgroup_table_publishes_n_and_refuses_no_difference() -> None:
    frame = pd.DataFrame(
        {
            "sex": [1.0, 1.0, 2.0, np.nan],
            "age": [30, 70, 50, np.nan],
            "bmi": [22, 31, 27, np.nan],
            "body_type": [0.0, 1.0, 2.0, np.nan],
            "has_previous_surgery": [0.0, 1.0, 0.0, np.nan],
        }
    )
    errors = pd.DataFrame({"endpoint_error_mean_mm": [1.0, 2.0, 8.0, 3.0]})
    table = subgroup_error_table(frame, errors)
    assert table["no_difference_claims"] is False
    sex_rows = {row["level"]: row for row in table["groups"]["sex"]}
    assert sex_rows["male"]["n"] == 2
    assert sex_rows["male"]["underpowered"] is True
    assert "wide" in sex_rows["male"]["inference_note"].lower() or "small" in sex_rows["male"]["inference_note"].lower()
