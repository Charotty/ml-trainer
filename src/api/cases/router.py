"""FastAPI router for CT Workbench cases."""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .features_service import build_features_from_base, merge_base_features
from .predictor import ProductionPredictor
from .report_service import DISCLAIMER, build_report_dict
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

from src.visualization.displacement_plots import build_case_report_pdf  # noqa: E402

_MODEL_UNAVAILABLE = (
    "Модель не загружена. Обучите clinical_honest.pkl и перезапустите CT Workbench API."
)


def create_cases_router(
    storage: CaseStorage,
    get_predictor: Callable[[], ProductionPredictor],
) -> APIRouter:
    router = APIRouter(prefix="/api/v1/cases", tags=["cases"])

    def _require_predictor() -> ProductionPredictor:
        try:
            return get_predictor()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=_MODEL_UNAVAILABLE) from exc

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
        predictor = _require_predictor()
        if not start_analyze(storage, case_id, predictor):
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
        pred = _require_predictor()
        base_out, all_features, coverage, missing = build_features_from_base(
            merged,
            feature_names=list(pred.payload["feature_names"]),
            enrichment_mode=pred.enrichment_mode(),
            na_trend_store=pred.payload.get("na_trend_store"),
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
        pred = _require_predictor()
        predictions = pred.predict_row(base)
        from .predictor import assess_prediction_sanity

        sanity_ok, warnings = assess_prediction_sanity(predictions)
        storage.write_json_artifact(
            case_id,
            "prediction.json",
            {
                "predictions": predictions,
                "model_id": pred.model_path.name,
                "enrichment_mode": pred.enrichment_mode(),
                "feature_count": pred.feature_count(),
                "sanity_ok": sanity_ok,
                "warnings": warnings,
            },
        )
        storage.update_meta(case_id, status="predicted")
        return PredictResponse(
            predictions=predictions,
            model_id=pred.model_path.name,
            enrichment_mode=pred.enrichment_mode(),
            feature_count=pred.feature_count(),
            sanity_ok=sanity_ok,
            warnings=warnings,
        )

    @router.get("/{case_id}/prediction")
    def get_prediction(case_id: str) -> Dict[str, Any]:
        try:
            storage.get_meta(case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        prediction = storage.read_json_artifact(case_id, "prediction.json")
        if not prediction:
            raise HTTPException(status_code=404, detail="No prediction yet. Run predict first.")
        return prediction

    @router.get("/{case_id}/report.json")
    def report_json(case_id: str) -> Dict[str, Any]:
        try:
            storage.get_meta(case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        report = build_report_dict(storage, case_id, features_view=get_features(case_id))
        storage.write_json_artifact(case_id, "report.json", report)
        storage.update_meta(case_id, status="reported")
        return report

    @router.get("/{case_id}/report.pdf")
    def report_pdf(case_id: str) -> FileResponse:
        try:
            storage.get_meta(case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        prediction = storage.read_json_artifact(case_id, "prediction.json")
        if not prediction:
            raise HTTPException(status_code=400, detail="No prediction yet. Run predict first.")
        report = build_report_dict(storage, case_id, features_view=get_features(case_id))
        storage.write_json_artifact(case_id, "report.json", report)
        pdf_path = storage.artifacts_dir(case_id) / "report.pdf"
        build_case_report_pdf(report, pdf_path)
        storage.update_meta(case_id, status="reported")
        label = report.get("meta", {}).get("patient_label") or case_id[:8]
        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(label))
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"ct_workbench_report_{safe_label}.pdf",
        )

    return router
