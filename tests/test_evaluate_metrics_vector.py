"""Unit tests for clinical within_5/10 vector metrics (audit stage 4)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
VALIDATION = ROOT / "scripts" / "validation"
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


def test_within_ratios_use_vector_error_mean_not_pointwise() -> None:
    # Patient 0: exact match → vector_error_mean = 0 (within 5 and 10)
    # Patient 1: |L2_true - L2_pred| = 6 on both sides → mean = 6 (outside 5, inside 10)
    # Patient 2: mean = 12 (outside 10)
    y_true = _rows(
        (3.0, 4.0, 0.0, 0.0, 0.0, 5.0),  # L=5, R=5
        (3.0, 4.0, 0.0, 0.0, 0.0, 5.0),  # L=5, R=5
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # L=0, R=0
    )
    y_pred = _rows(
        (3.0, 4.0, 0.0, 0.0, 0.0, 5.0),  # err 0
        (11.0, 0.0, 0.0, 11.0, 0.0, 0.0),  # L=11,R=11 → |5-11|=6
        (12.0, 0.0, 0.0, 12.0, 0.0, 0.0),  # L=12,R=12 → err 12
    )

    summary, per_patient = compute_clinical_within_ratios(y_true, y_pred, TARGET_COLUMNS)

    np.testing.assert_allclose(per_patient["vector_error_mean_mm"], [0.0, 6.0, 12.0])
    assert summary["within_5mm_ratio"] == pytest.approx(1.0 / 3.0)
    assert summary["within_10mm_ratio"] == pytest.approx(2.0 / 3.0)

    # Pointwise axis cells stay mostly small; clinical vector rate is stricter here
    assert summary["within_5mm_pointwise_ratio"] > summary["within_5mm_ratio"]


def test_perfect_predictions_are_fully_within() -> None:
    y = _rows((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (-1.0, 0.5, 2.0, 3.0, -2.0, 1.0))
    summary, per_patient = compute_clinical_within_ratios(y, y, TARGET_COLUMNS)
    assert np.allclose(per_patient["vector_error_mean_mm"], 0.0)
    assert summary["within_5mm_ratio"] == 1.0
    assert summary["within_10mm_ratio"] == 1.0
    assert summary["within_5mm_pointwise_ratio"] == 1.0
