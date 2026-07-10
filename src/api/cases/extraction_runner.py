"""Run DICOM extraction in an isolated process (native crashes must not kill the API)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.dicom_prep import make_ascii_work_slug  # noqa: E402
from scripts.inference.extract_from_dicom import process_case  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated case extraction worker")
    parser.add_argument("--dicom-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--reuse-nifti", action="store_true")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_slug = make_ascii_work_slug(args.case_id, index=1)
    nifti_dir = work_dir / f"nifti_{work_slug}"
    reuse_nifti = args.reuse_nifti or bool(list(nifti_dir.glob("*.nii*")))

    row = process_case(
        Path(args.dicom_dir),
        work_dir=work_dir,
        case_id=args.case_id,
        canonical=True,
        fast=args.fast,
        device=args.device,
        keep_temp=True,
        reuse_nifti=reuse_nifti,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    status = str(row.get("status") or "")
    if status in {"extracted", "partial", "metadata_only"}:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
