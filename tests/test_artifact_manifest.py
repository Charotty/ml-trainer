"""Focused tests for artifact manifest dump/validate (Block 0 provenance)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.artifact_manifest import (
    ARTIFACT_SCHEMA_VERSION,
    HISTORICAL_OOF_MAE_MM_DO_NOT_MIX,
    ManifestValidationError,
    OOF_METRICS_DO_NOT_MIX_NOTE,
    apply_oof_to_manifest,
    build_manifest,
    dump_manifest,
    extract_from_payload,
    load_manifest,
    oof_placeholder,
    validate_manifest,
)

FIXTURE = ROOT / "tests" / "fixtures" / "artifact_manifest_minimal.json"


def test_fixture_manifest_validates() -> None:
    card = load_manifest(FIXTURE)
    assert card["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert card["feature_count"] == len(card["feature_names"]) == 2
    assert card["production_winner"] is False
    assert "8.40" in card["oof_metrics_note"]
    assert "MUST NOT" in card["oof_metrics_note"]
    validate_manifest(card)


def test_missing_required_field_rejected() -> None:
    card = load_manifest(FIXTURE)
    del card["sha256"]
    with pytest.raises(ManifestValidationError, match="Missing required fields"):
        validate_manifest(card)


def test_feature_count_mismatch_rejected() -> None:
    card = load_manifest(FIXTURE)
    card["feature_count"] = 99
    with pytest.raises(ManifestValidationError, match="feature_count"):
        validate_manifest(card)


def test_oof_disclaimer_required() -> None:
    card = load_manifest(FIXTURE)
    card["oof_metrics_note"] = "looks good, MAE is fine"
    with pytest.raises(ManifestValidationError, match="8.40"):
        validate_manifest(card)


def test_dump_load_roundtrip(tmp_path: Path) -> None:
    payload_fields = extract_from_payload(
        {
            "feature_names": ["a", "b"],
            "target_names": ["kidney_left_delta_x"],
            "models": {},
            "training_meta": {"clinical_only": True},
            "left_z_calibrator": None,
            "right_z_calibrator": {"kind": "identity"},
        }
    )
    card = build_manifest(
        candidate_id="roundtrip",
        artifact_filename="roundtrip.pkl",
        sha256="BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
        size_bytes=1,
        payload_fields=payload_fields,
        git_commit="abc123",
        python_version="3.12.10",
        sklearn_version="1.9.0",
        run_id="PLACEHOLDER",
    )
    assert payload_fields["calibrators_present"]["right_z_calibrator"] is True
    assert payload_fields["calibrators_present"]["left_z_calibrator"] is False
    assert card["oof_metrics_note"] == OOF_METRICS_DO_NOT_MIX_NOTE
    assert card["historical_oof_mae_mm_do_not_mix"] == list(HISTORICAL_OOF_MAE_MM_DO_NOT_MIX)

    out = tmp_path / "card.json"
    dump_manifest(out, card)
    loaded = load_manifest(out)
    assert loaded["candidate_id"] == "roundtrip"
    assert loaded["feature_names"] == ["a", "b"]


def test_unsupported_schema_version_rejected() -> None:
    card = load_manifest(FIXTURE)
    card = copy.deepcopy(card)
    card["artifact_schema_version"] = "9.9.9"
    card["schema_version"] = "9.9.9"
    with pytest.raises(ManifestValidationError, match="Unsupported artifact_schema_version"):
        validate_manifest(card)


def test_fixture_is_json_object() -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    assert raw["size_bytes"] == 128


def test_oof_placeholder_does_not_require_retrain() -> None:
    card = load_manifest(FIXTURE)
    apply_oof_to_manifest(card, oof_placeholder(protocol="nested_groupkfold"))
    assert card["oof_status"] == "not_computed"
    assert card["folds"] is None
    validate_manifest(card)
