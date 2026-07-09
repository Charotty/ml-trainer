"""FastAPI router for CT Workbench cases."""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile

from .features_service import build_features_from_base, merge_base_features
from .predictor import ProductionPredictor
from .schemas import (
    CaseListResponse,
    CaseStatusResponse,
    CaseSummary,
    CreateCaseRequest,
    CreateCaseResponse,
    FeaturesResponse,
    ManualFeaturesPatch,
    PredictResponse,
)
from .storage import CaseStorage
from .worker import is_analyze_running, start_analyze

DISCLAIMER = (
    "Исследовательский инструмент. Не заменяет клинический протокол. "
    "Прогноз основан на supine-МСКТ и production-модели Adaptive Ensemble (na_trends)."
)


def create_cases_router(
    storage: CaseStorage,
    get_predictor: Callable[[], ProductionPredictor],
) -> APIRouter:
    router = APIRouter(prefix="/api/v1/cases", tags=["cases"])

    @router.post("", response_model=CreateCaseResponse)
    def create_case(body: CreateCaseRequest) -> CreateCaseResponse:
        meta = storage.create_case(patient_label=body.patient_label)
        return CreateCaseResponse(case_id=meta["case_id"], status=meta["status"])

    @router.get("", response_model=CaseListResponse)
    def list_cases() -> CaseListResponse:
        rows = storage.list_cases()
        summaries = [
            CaseSummary(
                case_id=r["case_id"],
                patient_label=r.get("patient_label"),
                status=r.get("status", "created"),
                created_at=r.get("created_at", ""),
                updated_at=r.get("updated_at", ""),
                coverage_pct=r.get("coverage_pct"),
            )
            for r in rows
        ]
        return CaseListResponse(cases=summaries, total=len(summaries))

    @router.post("/{case_id}/upload")
    async def upload_dicom(case_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
        try:
            storage.get_meta(case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        suffix = Path(file.filename or "upload.zip").suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".zip") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
        try:
            if suffix == ".zip":
                count = storage.save_upload_zip(case_id, tmp_path)
            else:
                dicom_dir = storage.dicom_dir(case_id)
                dicom_dir.mkdir(parents=True, exist_ok=True)
                target = dicom_dir / (file.filename or "upload.dcm")
                shutil.copy(tmp_path, target)
                count = 1
                storage.update_meta(case_id, status="uploaded", progress_pct=10.0)
            return {"case_id": case_id, "status": "uploaded", "files_saved": count}
        finally:
            tmp_path.unlink(missing_ok=True)

    @router.post("/{case_id}/analyze")
    def analyze_case(case_id: str) -> Dict[str, Any]:
        try:
            storage.get_meta(case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if is_analyze_running(case_id):
            return {"case_id": case_id, "status": "extracting", "message": "Job already running"}
        if not start_analyze(storage, case_id, get_predictor()):
            raise HTTPException(status_code=409, detail="Analyze already running")
        return {"case_id": case_id, "status": "extracting", "message": "Job started"}

    @router.get("/{case_id}/status", response_model=CaseStatusResponse)
    def case_status(case_id: str) -> CaseStatusResponse:
        try:
            meta = storage.get_meta(case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return CaseStatusResponse(
            case_id=meta["case_id"],
            status=meta.get("status", "created"),
            progress_pct=float(meta.get("progress_pct") or 0.0),
            stage=meta.get("stage"),
            message=meta.get("message"),
            error=meta.get("error"),
        )

    @router.get("/{case_id}/features", response_model=FeaturesResponse)
    def get_features(case_id: str) -> FeaturesResponse:
        try:
            storage.get_meta(case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        base = storage.read_json_artifact(case_id, "base_features.json") or {}
        feat_doc = storage.read_json_artifact(case_id, "features.json") or {}
        overrides_path = storage.artifacts_dir(case_id) / "manual_overrides.json"
        overrides = []
        if overrides_path.exists():
            import json

            overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
        return FeaturesResponse(
            base_features=base,
            all_features=feat_doc.get("all_features", {}),
            coverage_pct=float(feat_doc.get("coverage_pct") or 0.0),
            missing_features=list(feat_doc.get("missing_features") or []),
            manual_overrides=overrides,
        )

    @router.patch("/{case_id}/features/manual", response_model=FeaturesResponse)
    def patch_manual_features(case_id: str, body: ManualFeaturesPatch) -> FeaturesResponse:
        try:
            storage.get_meta(case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        base = storage.read_json_artifact(case_id, "base_features.json") or {}
        merged = merge_base_features(base, body.overrides)
        storage.append_manual_override(
            case_id,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "author": body.author,
                "reason": body.reason,
                "overrides": body.overrides,
            },
        )
        base_out, all_features, coverage, missing = build_features_from_base(
            merged,
            feature_names=list(predictor.payload["feature_names"]),
            enrichment_mode=predictor.enrichment_mode(),
            na_trend_store=predictor.payload.get("na_trend_store"),
        )
        storage.write_json_artifact(case_id, "base_features.json", base_out)
        storage.write_json_artifact(
            case_id,
            "features.json",
            {"all_features": all_features, "coverage_pct": coverage, "missing_features": missing},
        )
        storage.update_meta(case_id, status="qa_pending", coverage_pct=coverage)
        return get_features(case_id)

    @router.post("/{case_id}/predict", response_model=PredictResponse)
    def predict_case(case_id: str) -> PredictResponse:
        try:
            storage.get_meta(case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        base = storage.read_json_artifact(case_id, "base_features.json")
        if not base:
            raise HTTPException(status_code=400, detail="No features. Run analyze or manual input first.")
        pred = get_predictor()
        predictions = pred.predict_row(base)
        storage.write_json_artifact(
            case_id,
            "prediction.json",
            {
                "predictions": predictions,
                "model_id": pred.model_path.name,
                "enrichment_mode": pred.enrichment_mode(),
                "feature_count": pred.feature_count(),
            },
        )
        storage.update_meta(case_id, status="predicted")
        return PredictResponse(
            predictions=predictions,
            model_id=pred.model_path.name,
            enrichment_mode=pred.enrichment_mode(),
            feature_count=pred.feature_count(),
        )

    @router.get("/{case_id}/report.json")
    def report_json(case_id: str) -> Dict[str, Any]:
        try:
            meta = storage.get_meta(case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        report = {
            "schema_version": "ct_workbench_report_v1",
            "disclaimer": DISCLAIMER,
            "meta": meta,
            "base_features": storage.read_json_artifact(case_id, "base_features.json"),
            "features": storage.read_json_artifact(case_id, "features.json"),
            "prediction": storage.read_json_artifact(case_id, "prediction.json"),
            "manual_overrides": get_features(case_id).manual_overrides,
        }
        storage.write_json_artifact(case_id, "report.json", report)
        storage.update_meta(case_id, status="reported")
        return report

    return router
