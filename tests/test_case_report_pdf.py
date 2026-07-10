"""Unit tests for case PDF report charts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.displacement_plots import (  # noqa: E402
    build_case_report_pdf,
    quality_checks,
)


def test_quality_checks_pass_for_typical_prediction() -> None:
    left = np.array([12.0, -3.0, 2.0])
    right = np.array([-10.0, 4.0, -1.0])
    checks = quality_checks(left, right)
    assert checks["all_passed"] is True


def test_build_case_report_pdf_writes_file(tmp_path: Path) -> None:
    report = {
        "disclaimer": "Research only.",
        "meta": {
            "case_id": "case-1",
            "patient_label": "demo",
            "updated_at": "2026-07-10T12:00:00+00:00",
            "status": "predicted",
        },
        "extraction": {"status": "extracted", "totalsegmentator_status": "failed", "series_description": "Body", "series_slices": 350},
        "base_features": {
            "spine_center_x": 0.0,
            "spine_center_y": 0.0,
            "spine_center_z": 100.0,
            "body_width_mm": 280.0,
        },
        "features": {"coverage_pct": 45.0, "missing_features": ["kidney_left_volume_cm3"]},
        "prediction": {
            "predictions": {
                "kidney_left_delta_x": 8.0,
                "kidney_left_delta_y": -2.0,
                "kidney_left_delta_z": 3.0,
                "kidney_right_delta_x": -7.0,
                "kidney_right_delta_y": 1.5,
                "kidney_right_delta_z": -2.0,
            },
            "model_id": "mock.pkl",
            "enrichment_mode": "na_trends",
            "feature_count": 111,
        },
        "manual_overrides": {},
    }
    out = tmp_path / "report.pdf"
    build_case_report_pdf(report, out)
    assert out.exists()
    assert out.read_bytes()[:4] == b"%PDF"
