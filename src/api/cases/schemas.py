"""Pydantic schemas for Cases API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CreateCaseRequest(BaseModel):
    patient_label: Optional[str] = Field(None, description="Optional de-identified label")


class CreateCaseResponse(BaseModel):
    case_id: str
    status: str


class CaseStatusResponse(BaseModel):
    case_id: str
    status: str
    progress_pct: float = 0.0
    stage: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


class ManualFeaturesPatch(BaseModel):
    overrides: Dict[str, float] = Field(..., description="BASE_FEATURES overrides")
    reason: Optional[str] = Field(None, description="Why manual correction was made")
    author: Optional[str] = Field("local_user", description="Local user id for audit")


class FeaturesResponse(BaseModel):
    base_features: Dict[str, Any]
    all_features: Dict[str, Any]
    coverage_pct: float
    missing_features: List[str]
    manual_overrides: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: Dict[str, float]
    model_id: str
    enrichment_mode: str
    feature_count: int
    sanity_ok: bool = True
    warnings: List[str] = Field(default_factory=list)


class CaseSummary(BaseModel):
    case_id: str
    patient_label: Optional[str]
    status: str
    created_at: str
    updated_at: str
    coverage_pct: Optional[float] = None


class CaseListResponse(BaseModel):
    cases: List[CaseSummary]
    total: int
