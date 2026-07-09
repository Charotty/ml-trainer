#!/usr/bin/env python3
"""CT Workbench API — Cases REST + static frontend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.cases.predictor import ProductionPredictor
from src.api.cases.router import create_cases_router
from src.api.cases.storage import CaseStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_PUBLIC = REPO_ROOT / "frontend" / "public"

_storage = CaseStorage()
_predictor: Optional[ProductionPredictor] = None


def get_predictor() -> ProductionPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ProductionPredictor.load()
    return _predictor


def create_app(
    storage: Optional[CaseStorage] = None,
    predictor_factory: Optional[Callable[[], ProductionPredictor]] = None,
) -> FastAPI:
    store = storage or _storage
    pred_fn = predictor_factory or get_predictor

    application = FastAPI(
        title="CT Workbench API",
        description="DICOM upload, feature extraction, and kidney displacement prediction",
        version="0.1.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.on_event("startup")
    def _load_model() -> None:
        global _predictor
        if predictor_factory is not None:
            return
        try:
            _predictor = ProductionPredictor.load()
            logger.info(
                "Loaded model %s (%s features)",
                _predictor.model_path.name,
                _predictor.feature_count(),
            )
        except FileNotFoundError as exc:
            logger.warning("Model not loaded at startup: %s", exc)
            _predictor = None

    @application.get("/health")
    def health() -> dict:
        try:
            pred = pred_fn()
            loaded = pred is not None
            return {
                "status": "ok",
                "model_loaded": loaded,
                "model_id": pred.model_path.name if loaded else None,
                "feature_count": pred.feature_count() if loaded else 0,
            }
        except FileNotFoundError:
            return {"status": "ok", "model_loaded": False, "model_id": None, "feature_count": 0}

    application.include_router(create_cases_router(store, pred_fn))

    @application.get("/")
    def index() -> FileResponse:
        index_path = FRONTEND_PUBLIC / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return FileResponse(REPO_ROOT / "frontend" / "README.md")

    if FRONTEND_PUBLIC.exists():
        application.mount("/static", StaticFiles(directory=str(FRONTEND_PUBLIC)), name="static")

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.ct_workbench_api:app", host="127.0.0.1", port=8010, reload=True)
