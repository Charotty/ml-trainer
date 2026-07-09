"""Filesystem storage for CT Workbench cases."""

from __future__ import annotations

import json
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
CASES_ROOT = REPO_ROOT / "data" / "cases"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CaseStorage:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or CASES_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)

    def case_dir(self, case_id: str) -> Path:
        return self.root / case_id

    def create_case(self, patient_label: Optional[str] = None) -> Dict[str, Any]:
        case_id = str(uuid.uuid4())
        case_path = self.case_dir(case_id)
        for sub in ("dicom", "artifacts", "logs"):
            (case_path / sub).mkdir(parents=True, exist_ok=True)
        meta = {
            "case_id": case_id,
            "patient_label": patient_label,
            "status": "created",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "progress_pct": 0.0,
            "stage": None,
            "message": None,
            "error": None,
        }
        self._write_meta(case_id, meta)
        return meta

    def list_cases(self) -> List[Dict[str, Any]]:
        cases: List[Dict[str, Any]] = []
        if not self.root.exists():
            return cases
        for path in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not path.is_dir():
                continue
            meta_path = path / "meta.json"
            if meta_path.exists():
                cases.append(json.loads(meta_path.read_text(encoding="utf-8")))
        return cases

    def get_meta(self, case_id: str) -> Dict[str, Any]:
        path = self.case_dir(case_id) / "meta.json"
        if not path.exists():
            raise FileNotFoundError(f"Case not found: {case_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def update_meta(self, case_id: str, **fields: Any) -> Dict[str, Any]:
        meta = self.get_meta(case_id)
        meta.update(fields)
        meta["updated_at"] = _utc_now()
        self._write_meta(case_id, meta)
        return meta

    def _write_meta(self, case_id: str, meta: Dict[str, Any]) -> None:
        path = self.case_dir(case_id) / "meta.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_upload_zip(self, case_id: str, zip_path: Path) -> int:
        dicom_dir = self.case_dir(case_id) / "dicom"
        if dicom_dir.exists():
            shutil.rmtree(dicom_dir)
        dicom_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                target = dicom_dir / Path(name).name
                target.write_bytes(zf.read(name))
                count += 1
        self.update_meta(
            case_id,
            status="uploaded",
            progress_pct=10.0,
            stage="uploaded",
            message=f"Saved {count} files",
        )
        return count

    def dicom_dir(self, case_id: str) -> Path:
        return self.case_dir(case_id) / "dicom"

    def artifacts_dir(self, case_id: str) -> Path:
        return self.case_dir(case_id) / "artifacts"

    def read_json_artifact(self, case_id: str, name: str) -> Optional[Dict[str, Any]]:
        path = self.artifacts_dir(case_id) / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def write_json_artifact(self, case_id: str, name: str, payload: Dict[str, Any]) -> Path:
        path = self.artifacts_dir(case_id) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def append_manual_override(self, case_id: str, entry: Dict[str, Any]) -> None:
        path = self.artifacts_dir(case_id) / "manual_overrides.json"
        rows: List[Dict[str, Any]] = []
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))
        rows.append(entry)
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    def log(self, case_id: str, line: str) -> None:
        log_path = self.case_dir(case_id) / "logs" / "case.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{_utc_now()}] {line}\n")
