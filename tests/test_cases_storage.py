"""Tests for case filesystem storage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.api.cases.storage import CaseStorage


@pytest.fixture
def storage(tmp_path: Path) -> CaseStorage:
    return CaseStorage(root=tmp_path / "cases")


def test_create_and_list_case(storage: CaseStorage) -> None:
    meta = storage.create_case(patient_label="test-1")
    case_id = meta["case_id"]
    assert meta["status"] == "created"
    listed = storage.list_cases()
    assert len(listed) == 1
    assert listed[0]["case_id"] == case_id


def test_update_meta(storage: CaseStorage) -> None:
    meta = storage.create_case()
    case_id = meta["case_id"]
    storage.update_meta(case_id, status="uploaded", progress_pct=50.0)
    updated = storage.get_meta(case_id)
    assert updated["status"] == "uploaded"
    assert updated["progress_pct"] == 50.0


def test_json_artifacts(storage: CaseStorage) -> None:
    meta = storage.create_case()
    case_id = meta["case_id"]
    storage.write_json_artifact(case_id, "base_features.json", {"kidney_left_center_x_rel": 1.0})
    loaded = storage.read_json_artifact(case_id, "base_features.json")
    assert loaded is not None
    assert loaded["kidney_left_center_x_rel"] == 1.0


def test_manual_override_append(storage: CaseStorage) -> None:
    meta = storage.create_case()
    case_id = meta["case_id"]
    storage.append_manual_override(case_id, {"overrides": {"a": 1}})
    storage.append_manual_override(case_id, {"overrides": {"b": 2}})
    path = storage.artifacts_dir(case_id) / "manual_overrides.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert len(rows) == 2
