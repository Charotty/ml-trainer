"""Background extraction jobs for cases."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .features_service import build_features_from_base, extraction_row_to_base_features  # noqa: E402
from .predictor import ProductionPredictor  # noqa: E402
from .storage import CaseStorage  # noqa: E402

_EXTRACTION_TIMEOUT_SEC = 3600

_lock = threading.Lock()
_running: set[str] = set()


def is_analyze_running(case_id: str) -> bool:
    with _lock:
        return case_id in _running


def run_analyze_job(
    storage: CaseStorage,
    case_id: str,
    predictor: ProductionPredictor,
    *,
    fast: bool = True,
) -> None:
    with _lock:
        if case_id in _running:
            return
        _running.add(case_id)

    def _job() -> None:
        try:
            storage.update_meta(
                case_id,
                status="extracting",
                progress_pct=15.0,
                stage="prepare",
                message="Preparing DICOM series",
                error=None,
            )
            storage.log(case_id, "analyze job started")
            dicom_dir = storage.dicom_dir(case_id)
            if not any(dicom_dir.iterdir()):
                raise FileNotFoundError("No DICOM files in case folder")

            work_dir = storage.artifacts_dir(case_id) / "work"
            work_dir.mkdir(parents=True, exist_ok=True)
            storage.update_meta(case_id, progress_pct=30.0, stage="segmentation", message="Running extraction")

            result_path = work_dir / "extraction_result.json"
            cmd = [
                sys.executable,
                "-m",
                "src.api.cases.extraction_runner",
                "--dicom-dir",
                str(dicom_dir),
                "--work-dir",
                str(work_dir),
                "--case-id",
                case_id,
                "--output",
                str(result_path),
                "--device",
                "cpu",
            ]
            if fast:
                cmd.append("--fast")
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=_EXTRACTION_TIMEOUT_SEC,
            )
            if proc.returncode != 0:
                detail = ""
                if result_path.exists():
                    row = json.loads(result_path.read_text(encoding="utf-8"))
                    err = row.get("error") or detail or f"exit code {proc.returncode}"
                    raise RuntimeError(str(err))
                if proc.returncode < 0 or proc.returncode > 255:
                    raise RuntimeError(
                        "Extraction process crashed during segmentation "
                        f"(exit {proc.returncode}). {detail}".strip()
                    )
                raise RuntimeError(detail or f"Extraction failed with exit code {proc.returncode}")
            row = json.loads(result_path.read_text(encoding="utf-8"))
            storage.write_json_artifact(case_id, "extraction_raw.json", row)
            storage.log(case_id, f"extraction status={row.get('status')}")

            storage.update_meta(case_id, progress_pct=70.0, stage="features", message="Building feature vector")
            base_row = extraction_row_to_base_features(row)
            base_out, all_features, coverage, missing = build_features_from_base(
                base_row,
                feature_names=list(predictor.payload["feature_names"]),
                enrichment_mode=predictor.enrichment_mode(),
                na_trend_store=predictor.payload.get("na_trend_store"),
            )
            storage.write_json_artifact(case_id, "base_features.json", base_out)
            storage.write_json_artifact(
                case_id,
                "features.json",
                {
                    "all_features": all_features,
                    "coverage_pct": coverage,
                    "missing_features": missing,
                },
            )
            storage.update_meta(
                case_id,
                status="features_ready",
                progress_pct=100.0,
                stage="done",
                message=f"Features ready (coverage {coverage:.1f}%)",
                coverage_pct=coverage,
            )
            storage.log(case_id, "analyze job completed")
        except Exception as exc:
            storage.update_meta(
                case_id,
                status="failed",
                progress_pct=0.0,
                stage="error",
                message="Extraction failed",
                error=str(exc),
            )
            storage.log(case_id, f"analyze job failed: {exc}")
        finally:
            with _lock:
                _running.discard(case_id)

    thread = threading.Thread(target=_job, daemon=True)
    thread.start()


def start_analyze(storage: CaseStorage, case_id: str, predictor: ProductionPredictor) -> bool:
    if is_analyze_running(case_id):
        return False
    run_analyze_job(storage, case_id, predictor)
    return True
