"""API tests for CT Workbench cases (no DICOM extraction)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.cases.storage import CaseStorage
from src.api.ct_workbench_api import create_app


def _mock_predictor():
    mock = MagicMock()
    mock.payload = {
        "feature_names": ["kidney_left_center_x_rel", "body_width_mm"],
        "enrichment_mode": "na_trends",
        "na_trend_store": None,
    }
    mock.model_path = Path("mock.pkl")
    mock.enrichment_mode.return_value = "na_trends"
    mock.feature_count.return_value = 2
    mock.predict_row.return_value = {
        "kidney_left_delta_x": 1.0,
        "kidney_left_delta_y": 2.0,
        "kidney_left_delta_z": 3.0,
        "kidney_right_delta_x": 4.0,
        "kidney_right_delta_y": 5.0,
        "kidney_right_delta_z": 6.0,
    }
    return mock


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    storage = CaseStorage(root=tmp_path / "cases")
    application = create_app(storage=storage, predictor_factory=_mock_predictor)
    return TestClient(application)


def test_create_case_and_predict_flow(client: TestClient, tmp_path: Path) -> None:
    create = client.post("/api/v1/cases", json={"patient_label": "p1"})
    assert create.status_code == 200
    case_id = create.json()["case_id"]

    storage = CaseStorage(root=tmp_path / "cases")
    base = {
        "kidney_left_center_x_rel": 10.0,
        "kidney_left_center_y_rel": 20.0,
        "kidney_left_center_z_rel": 30.0,
        "kidney_right_center_x_rel": 11.0,
        "kidney_right_center_y_rel": 21.0,
        "kidney_right_center_z_rel": 31.0,
        "body_width_mm": 300.0,
        "body_depth_mm": 200.0,
    }
    storage.write_json_artifact(case_id, "base_features.json", base)
    storage.write_json_artifact(
        case_id,
        "features.json",
        {"all_features": base, "coverage_pct": 90.0, "missing_features": []},
    )
    storage.update_meta(case_id, status="features_ready")

    feat = client.get(f"/api/v1/cases/{case_id}/features")
    assert feat.status_code == 200

    pred = client.post(f"/api/v1/cases/{case_id}/predict")
    assert pred.status_code == 200
    body = pred.json()
    assert body["predictions"]["kidney_left_delta_z"] == 3.0
    assert body["sanity_ok"] is True
    assert body["warnings"] == []

    report = client.get(f"/api/v1/cases/{case_id}/report.json")
    assert report.status_code == 200
    assert "disclaimer" in report.json()

    pdf = client.get(f"/api/v1/cases/{case_id}/report.pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content[:4] == b"%PDF"


def test_predict_returns_503_when_model_missing(tmp_path: Path) -> None:
    storage = CaseStorage(root=tmp_path / "cases")

    def _missing_model():
        raise FileNotFoundError("no pkl")

    application = create_app(storage=storage, predictor_factory=_missing_model)
    client = TestClient(application)
    create = client.post("/api/v1/cases", json={"patient_label": "p2"})
    case_id = create.json()["case_id"]
    storage.write_json_artifact(case_id, "base_features.json", {"kidney_left_center_x_rel": 1.0})
    storage.update_meta(case_id, status="features_ready")
    pred = client.post(f"/api/v1/cases/{case_id}/predict")
    assert pred.status_code == 503
    assert "модель" in pred.json()["detail"].lower() or "Модель" in pred.json()["detail"]


def test_analyze_returns_503_when_model_missing(tmp_path: Path) -> None:
    storage = CaseStorage(root=tmp_path / "cases")

    def _missing_model():
        raise FileNotFoundError("no pkl")

    application = create_app(storage=storage, predictor_factory=_missing_model)
    client = TestClient(application)
    create = client.post("/api/v1/cases", json={})
    case_id = create.json()["case_id"]
    storage.update_meta(case_id, status="uploaded")
    res = client.post(f"/api/v1/cases/{case_id}/analyze")
    assert res.status_code == 503
