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


def test_save_upload_zip_preserves_nested_same_basename(storage: CaseStorage, tmp_path: Path) -> None:
    """Nested ZIP members with identical basenames must not overwrite each other."""
    import zipfile

    meta = storage.create_case()
    case_id = meta["case_id"]
    zip_path = tmp_path / "series.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("series_a/IM0001.dcm", b"AAA")
        zf.writestr("series_b/IM0001.dcm", b"BBB")
    count = storage.save_upload_zip(case_id, zip_path)
    assert count == 2
    dicom_dir = storage.dicom_dir(case_id)
    a = dicom_dir / "series_a" / "IM0001.dcm"
    b = dicom_dir / "series_b" / "IM0001.dcm"
    assert a.exists() and b.exists()
    assert a.read_bytes() == b"AAA"
    assert b.read_bytes() == b"BBB"


def test_save_upload_zip_rejects_path_traversal(storage: CaseStorage, tmp_path: Path) -> None:
    import zipfile

    meta = storage.create_case()
    case_id = meta["case_id"]
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../outside.dcm", b"NOPE")
        zf.writestr("safe/ok.dcm", b"OK")
    count = storage.save_upload_zip(case_id, zip_path)
    assert count == 1
    assert (storage.dicom_dir(case_id) / "safe" / "ok.dcm").read_bytes() == b"OK"
    assert not (storage.root.parent / "outside.dcm").exists()
