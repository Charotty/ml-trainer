"""Unit tests for clinical 3D endpoint metrics (audit stage 4 / Block 4)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
VALIDATION = ROOT / "scripts" / "validation"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(VALIDATION) not in sys.path:
    sys.path.insert(0, str(VALIDATION))

from evaluate_metrics import compute_clinical_within_ratios  # noqa: E402

TARGET_COLUMNS = [
    "kidney_left_delta_x",
    "kidney_left_delta_y",
    "kidney_left_delta_z",
    "kidney_right_delta_x",
    "kidney_right_delta_y",
    "kidney_right_delta_z",
]


def _rows(*vectors: tuple[float, float, float, float, float, float]) -> np.ndarray:
    return np.asarray(vectors, dtype=float)


def test_within_ratios_use_3d_endpoint_error_not_pointwise() -> None:
    # Patient 0: exact match → 0 (within 5 and 10)
    # Patient 1: 6 mm Euclidean on both kidneys → mean 6 (outside 5, inside 10)
    # Patient 2: 12 mm on both → mean 12 (outside 10)
    y_true = _rows(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    y_pred = _rows(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (6.0, 0.0, 0.0, 6.0, 0.0, 0.0),
        (12.0, 0.0, 0.0, 12.0, 0.0, 0.0),
    )

    summary, per_patient = compute_clinical_within_ratios(y_true, y_pred, TARGET_COLUMNS)

    np.testing.assert_allclose(per_patient["endpoint_error_mean_mm"], [0.0, 6.0, 12.0])
    assert summary["within_5mm_ratio"] == pytest.approx(1.0 / 3.0)
    assert summary["within_10mm_ratio"] == pytest.approx(2.0 / 3.0)
    assert summary["within_15mm_ratio"] == pytest.approx(1.0)
    assert summary["p90_endpoint_error_mm"] >= 6.0


def test_perfect_predictions_are_fully_within() -> None:
    y = _rows((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (-1.0, 0.5, 2.0, 3.0, -2.0, 1.0))
    summary, per_patient = compute_clinical_within_ratios(y, y, TARGET_COLUMNS)
    assert np.allclose(per_patient["endpoint_error_mean_mm"], 0.0)
    assert summary["within_5mm_ratio"] == 1.0
    assert summary["within_10mm_ratio"] == 1.0
    assert summary["within_15mm_ratio"] == 1.0
    assert summary["within_5mm_pointwise_ratio"] == 1.0
