"""Assemble CT Workbench case reports (JSON + PDF)."""

from __future__ import annotations

from typing import Any, Dict

from .schemas import FeaturesResponse
from .storage import CaseStorage

DISCLAIMER = (
    "Исследовательский инструмент для планирования доступа. "
    "Не заменяет клинический протокол, осмотр и решение лечащего врача."
)


def build_report_dict(
    storage: CaseStorage,
    case_id: str,
    *,
    features_view: FeaturesResponse,
) -> Dict[str, Any]:
    meta = storage.get_meta(case_id)
    extraction = storage.read_json_artifact(case_id, "extraction_raw.json") or {}
    return {
        "schema_version": "ct_workbench_report_v1",
        "disclaimer": DISCLAIMER,
        "meta": meta,
        "extraction": extraction,
        "base_features": storage.read_json_artifact(case_id, "base_features.json"),
        "features": storage.read_json_artifact(case_id, "features.json"),
        "prediction": storage.read_json_artifact(case_id, "prediction.json"),
        "manual_overrides": features_view.manual_overrides,
    }
