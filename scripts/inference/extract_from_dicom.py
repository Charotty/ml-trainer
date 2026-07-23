#!/usr/bin/env python3
"""
DICOM preparation + feature extraction pipeline for amImageViewer / PACS exports.

Typical layout (F:/На Боку):
  Patient Name DD.MM.YYYY/
    DICOMDIR, Images.cds, amImageViewer.exe
    25111414/00000001 ...   # extensionless CT slices, often multiple series mixed

Nested archive (F:/На спине):
  2. Контроль 2023/
    .../PA000014/ST000001/SE000006/   # leaf series folders

Problems with scripts/inference/enhanced_ct_extractor.py on raw folders:
  - passes the whole patient tree to TotalSegmentator → dicom2nifti MISSING_DICOM_FILES
  - mixed series (scout + axial CT) in one directory
  - lightweight HU fallback masks TS failures

This script:
  1. discovers cases (flat or nested)
  2. selects the main CT series (pydicom grouping)
  3. converts via dcm2niix binary → NIfTI
  4. runs TotalSegmentator on NIfTI (kidneys)
  5. optionally runs enhanced_ct_extractor on the clean series folder (--canonical)

Usage:
  python scripts/inference/extract_from_dicom.py "F:/На Боку" --canonical --output results/boku.csv
  python scripts/inference/extract_from_dicom.py /path/to/patient --device auto --fast
  python scripts/inference/extract_from_dicom.py "F:/На Боку" --prep-only --temp-dir /tmp/dicom_prep
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.inference.dicom_prep import (  # noqa: E402
    PrepResult,
    cleanup_case_temp,
    discover_patient_cases,
    find_dcm2niix_executable,
    group_dicom_series,
    make_ascii_work_slug,
    prepare_case,
    resolve_totalsegmentator_device,
    select_main_ct_series,
)

try:
    import pydicom
except ImportError:
    print("ERROR: pydicom is required. pip install pydicom")
    sys.exit(1)

try:
    import nibabel as nib
except ImportError:
    nib = None  # type: ignore[assignment]

try:
    from totalsegmentator.python_api import totalsegmentator
except ImportError:
    totalsegmentator = None

try:
    from src.features.ct_geometry import kidney_features_from_mask
except ImportError:
    kidney_features_from_mask = None  # type: ignore[assignment,misc]

from scripts.inference.enhanced_ct_extractor import (  # noqa: E402
    _add_unified_features,
    _get_accuracy_params,
    _normalize_name,
    extract_features_from_dicom_folder,
)


def extract_dicom_metadata(dicom_file: Path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    try:
        dcm = pydicom.dcmread(str(dicom_file), stop_before_pixels=True, force=True)
        metadata["patient_id"] = str(dcm.PatientID) if "PatientID" in dcm else None
        metadata["patient_name"] = str(dcm.PatientName) if "PatientName" in dcm else None
        if "PatientSex" in dcm:
            # Match excel_displacement_adapter / CLINICAL_DEMOGRAPHIC_FEATURES: M=1, F=2
            sex = str(dcm.PatientSex).strip().upper()
            metadata["sex"] = 1.0 if sex == "M" else 2.0 if sex == "F" else None
        if "PatientAge" in dcm:
            age_str = str(dcm.PatientAge).replace("Y", "").replace("y", "")
            try:
                metadata["age"] = int(age_str)
            except ValueError:
                metadata["age"] = None
        metadata["weight_kg"] = float(dcm.PatientWeight) if "PatientWeight" in dcm else None
        metadata["height_m"] = float(dcm.PatientSize) if "PatientSize" in dcm else None
        if metadata.get("weight_kg") and metadata.get("height_m"):
            metadata["bmi"] = metadata["weight_kg"] / (metadata["height_m"] ** 2)
        metadata["patient_position"] = str(dcm.PatientPosition) if "PatientPosition" in dcm else None
        metadata["study_date"] = str(dcm.StudyDate) if "StudyDate" in dcm else None
        metadata["slice_thickness"] = float(dcm.SliceThickness) if "SliceThickness" in dcm else None
    except Exception as exc:
        metadata["metadata_error"] = str(exc)
    return metadata


def _has_segmentation_output(output_folder: Path) -> bool:
    if not output_folder.is_dir():
        return False
    for pattern in ("kidney_*.nii*", "segmentation.nii*"):
        if list(output_folder.glob(pattern)):
            return True
    return any(output_folder.iterdir())


def run_totalsegmentator(
    input_path: Path,
    output_folder: Path,
    *,
    fast: bool = False,
    device: str = "auto",
    roi_subset: Optional[Sequence[str]] = None,
) -> bool:
    if totalsegmentator is None:
        print("  TotalSegmentator not installed (pip install TotalSegmentator)")
        return False

    dev = resolve_totalsegmentator_device(device)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    kwargs = {
        "input": str(input_path),
        "output": str(output_folder),
        "nr_thr_resamp": 1,
        "nr_thr_saving": 1,
        "roi_subset": list(roi_subset or ["kidney_right", "kidney_left"]),
        "fast": fast,
        "quiet": True,
        "device": dev,
    }

    try:
        print(f"  TotalSegmentator device={dev} input={input_path.name}")
        totalsegmentator(**kwargs)
        if _has_segmentation_output(output_folder):
            return True
        print("  TotalSegmentator: no mask files written")
        return False
    except TypeError:
        kwargs.pop("roi_subset", None)
        totalsegmentator(**kwargs)
        if _has_segmentation_output(output_folder):
            return True
        print("  TotalSegmentator: no mask files written (fallback call)")
        return False
    except MemoryError as exc:
        if not fast:
            print(f"  memory error, retry --fast: {exc}")
            return run_totalsegmentator(
                input_path, output_folder, fast=True, device=device, roi_subset=roi_subset
            )
        print(f"  TotalSegmentator memory error: {exc}")
        return False
    except Exception as exc:
        print(f"  TotalSegmentator error: {exc}")
        return False


def _kidney_features_from_roi_mask(mask_path: Path, prefix: str) -> Dict[str, float]:
    if nib is None or kidney_features_from_mask is None:
        return {}
    try:
        img = nib.load(str(mask_path))
        data = img.get_fdata()
        mask = data > 0
        if not np.any(mask):
            return {}
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        return kidney_features_from_mask(mask, img.affine, zooms, prefix)
    except Exception as exc:
        print(f"  kidney mask parse error ({mask_path.name}): {exc}")
        return {}


def extract_kidney_features_from_ts_output(output_folder: Path) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for prefix, fname in (
        ("kidney_right", "kidney_right.nii.gz"),
        ("kidney_left", "kidney_left.nii.gz"),
    ):
        mask_path = output_folder / fname
        if mask_path.exists():
            result.update(_kidney_features_from_roi_mask(mask_path, prefix))

    seg_file = output_folder / "segmentation.nii.gz"
    if seg_file.exists() and nib is not None and kidney_features_from_mask is not None:
        try:
            seg_img = nib.load(str(seg_file))
            seg_data = seg_img.get_fdata()
            affine = seg_img.affine
            zooms = tuple(float(z) for z in seg_img.header.get_zooms()[:3])
            for label_id, prefix in ((8, "kidney_right"), (9, "kidney_left")):
                if any(k.startswith(prefix) for k in result):
                    continue
                mask = seg_data == label_id
                if np.any(mask):
                    result.update(kidney_features_from_mask(mask, affine, zooms, prefix))
        except Exception as exc:
            print(f"  combined segmentation parse error: {exc}")
    return result


def process_case(
    case_folder: Path,
    *,
    work_dir: Path,
    case_index: int = 1,
    use_totalsegmentator: bool = True,
    canonical: bool = False,
    accuracy_mode: str = "balanced",
    fast: bool = False,
    device: str = "auto",
    compress_nifti: bool = True,
    reuse_nifti: bool = False,
    prep_only: bool = False,
    roi_subset: Optional[Sequence[str]] = None,
    case_id: Optional[str] = None,
    keep_temp: bool = False,
) -> Dict[str, Any]:
    case_folder = Path(case_folder)
    case_id = case_id or case_folder.name
    work_slug = make_ascii_work_slug(case_id, index=case_index)
    print(f"\n[case] {case_id}  (work={work_slug})")

    try:
        return _process_case_inner(
            case_folder=case_folder,
            case_id=case_id,
            work_slug=work_slug,
            work_dir=work_dir,
            case_index=case_index,
            use_totalsegmentator=use_totalsegmentator,
            canonical=canonical,
            accuracy_mode=accuracy_mode,
            fast=fast,
            device=device,
            compress_nifti=compress_nifti,
            reuse_nifti=reuse_nifti,
            prep_only=prep_only,
            roi_subset=roi_subset,
        )
    finally:
        if not keep_temp:
            cleanup_case_temp(work_dir, work_slug)
        _release_gpu_memory()


def _release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _process_case_inner(
    *,
    case_folder: Path,
    case_id: str,
    work_slug: str,
    work_dir: Path,
    case_index: int,
    use_totalsegmentator: bool,
    canonical: bool,
    accuracy_mode: str,
    fast: bool,
    device: str,
    compress_nifti: bool,
    reuse_nifti: bool,
    prep_only: bool,
    roi_subset: Optional[Sequence[str]],
) -> Dict[str, Any]:
    prep = prepare_case(
        case_folder,
        work_dir,
        case_id=case_id,
        work_slug=work_slug,
        case_index=case_index,
        compress_nifti=compress_nifti,
        reuse_nifti=reuse_nifti,
    )

    row: Dict[str, Any] = {
        "case_id": case_id,
        "work_slug": prep.work_slug,
        "dicom_folder": str(case_folder),
        "series_uid": prep.series.series_uid,
        "series_description": prep.series.description,
        "series_slices": prep.series.slice_count,
        "series_input_dir": str(prep.series_input_dir),
        "nifti_input_file": str(prep.nifti_file) if prep.nifti_file else None,
        "prep_warnings": ";".join(prep.warnings) if prep.warnings else None,
        "dcm2niix_available": find_dcm2niix_executable() is not None,
    }

    if prep.series.files:
        row.update(extract_dicom_metadata(prep.series.files[0]))

    if prep_only:
        row["status"] = "prepared"
        return row

    kidney_feats: Dict[str, float] = {}
    ts_ok = False
    if use_totalsegmentator and prep.nifti_file:
        seg_dir = work_dir / f"seg_{prep.work_slug}"
        if seg_dir.exists():
            shutil.rmtree(seg_dir, ignore_errors=True)
        seg_dir.mkdir(parents=True, exist_ok=True)
        ts_ok = run_totalsegmentator(
            prep.nifti_file,
            seg_dir,
            fast=fast,
            device=device,
            roi_subset=roi_subset,
        )
        if ts_ok:
            kidney_feats = extract_kidney_features_from_ts_output(seg_dir)
            row["totalsegmentator_status"] = "ok" if kidney_feats else "empty_masks"
        else:
            row["totalsegmentator_status"] = "failed"
        gc.collect()
    elif use_totalsegmentator:
        row["totalsegmentator_status"] = "no_nifti"

    if canonical:
        params = _get_accuracy_params(accuracy_mode)
        try:
            feats = extract_features_from_dicom_folder(
                prep.series_input_dir,
                downsample=params["downsample"],
                max_slices=params["max_slices"],
                enable_kidney_segmentation=False,
                show_progress=False,
                slice_strategy=params["slice_strategy"],
            )
            feats = _add_unified_features(feats)
            row.update(feats)
            row["status"] = "extracted"
        except Exception as exc:
            print(f"  enhanced extractor error: {exc}")
            row["status"] = "partial" if kidney_feats else "error"
            row["error"] = str(exc)
        if kidney_feats:
            row.update(kidney_feats)
            row["kidney_source"] = "totalsegmentator_nifti"
    else:
        row.update(kidney_feats)
        for level in ("upper", "middle", "lower"):
            for side in ("left", "right"):
                for axis in ("x", "y", "z"):
                    key = f"kidney_{side}_{level}_{axis}"
                    legacy = f"{axis.upper()}_{level}_{side}"
                    if key in kidney_feats:
                        row[legacy] = kidney_feats[key]
        row["status"] = "extracted" if kidney_feats else "metadata_only"
        row["kidney_source"] = "totalsegmentator_nifti" if kidney_feats else None

    if canonical and not row.get("full_name_key"):
        row["full_name_key"] = _normalize_name(str(row.get("patient_name") or case_id))

    return row


def resolve_input_roots(paths: Sequence[str], dicom_root: Optional[Path]) -> List[Path]:
    roots: List[Path] = []
    if dicom_root:
        roots.append(Path(dicom_root))
    for p in paths:
        roots.append(Path(p))
    return roots


def collect_cases(
    roots: Sequence[Path],
    *,
    layout: str,
    max_cases: Optional[int],
) -> List[Tuple[Path, str]]:
    """Return ordered list of (case_folder, unique_case_id)."""
    cases: List[Tuple[Path, str]] = []
    seen_paths: set[str] = set()
    used_ids: set[str] = set()

    def add(case: Path) -> None:
        key = str(case.resolve())
        if key in seen_paths:
            return
        seen_paths.add(key)
        case_id = case.name
        if case_id in used_ids:
            suffix = 2
            while f"{case_id} ({suffix})" in used_ids:
                suffix += 1
            case_id = f"{case_id} ({suffix})"
        used_ids.add(case_id)
        cases.append((case, case_id))

    for root in roots:
        if root.is_file():
            add(root.parent)
            continue
        for case in discover_patient_cases(root, layout=layout):
            add(case)
            if max_cases and len(cases) >= max_cases:
                return cases
    return cases


def _needs_reprocess(row: Dict[str, Any]) -> bool:
    """A previously-saved row that should be recomputed (failed segmentation)."""
    status = str(row.get("status", "")).strip()
    ts = str(row.get("totalsegmentator_status", "")).strip()
    if status in ("error", "metadata_only", "", "nan"):
        return True
    if ts not in ("ok",):
        return True
    return False


def _load_existing_rows(out_path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """case_id -> row dict, plus original column order."""
    if not out_path.exists():
        return {}, []
    df = pd.read_csv(out_path)
    cols = list(df.columns)
    rows: Dict[str, Dict[str, Any]] = {}
    for rec in df.to_dict(orient="records"):
        cleaned = {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in rec.items()}
        rows[str(cleaned.get("case_id"))] = cleaned
    return rows, cols


def _write_rows(out_path: Path, rows_by_id: Dict[str, Dict[str, Any]], order: List[str], base_cols: List[str]) -> pd.DataFrame:
    ordered_rows = [rows_by_id[cid] for cid in order if cid in rows_by_id]
    df = pd.DataFrame(ordered_rows)
    if base_cols:
        extra = [c for c in df.columns if c not in base_cols]
        df = df.reindex(columns=base_cols + extra)
    # Atomic tmp+rename is preferred, but os.replace over an existing file on the
    # Windows DrvFs mount can raise PermissionError; fall back to a direct write.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, out_path)
    except (PermissionError, OSError):
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        df.to_csv(out_path, index=False)
    return df


def run_job(
    root: Path,
    out_path: Path,
    *,
    args: argparse.Namespace,
    work_dir: Path,
    prep_only: bool,
) -> None:
    print("\n" + "#" * 72)
    print(f"# JOB root={root}")
    print(f"#      output={out_path}")
    print("#" * 72)

    if not root.exists():
        print(f"  [skip] root not found: {root}")
        return

    cases = collect_cases([root], layout=args.layout, max_cases=args.max_cases)
    if not cases:
        print("  [skip] no cases found")
        return
    print(f"  cases discovered: {len(cases)}")

    existing_rows, base_cols = ({}, [])
    if args.update_existing:
        existing_rows, base_cols = _load_existing_rows(out_path)
        if existing_rows:
            print(f"  existing rows loaded: {len(existing_rows)} (update mode)")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_id: Dict[str, Dict[str, Any]] = dict(existing_rows)
    order: List[str] = list(existing_rows.keys())
    for _, cid in cases:
        if cid not in order:
            order.append(cid)

    processed = skipped = 0
    for i, (case_folder, case_id) in enumerate(cases, 1):
        prior = existing_rows.get(case_id)
        if args.update_existing and prior is not None and not _needs_reprocess(prior):
            skipped += 1
            continue

        print(f"\n[{i}/{len(cases)}]", end=" ")
        try:
            row = process_case(
                case_folder,
                work_dir=work_dir,
                case_index=i,
                case_id=case_id,
                use_totalsegmentator=not args.no_segmentation and not prep_only,
                canonical=args.canonical,
                accuracy_mode=args.accuracy_mode,
                fast=args.fast,
                device=args.device,
                compress_nifti=not args.no_compression,
                reuse_nifti=args.reuse_nifti,
                prep_only=prep_only,
                roi_subset=args.roi_subset,
                keep_temp=args.keep_temp,
            )
            print(f"  -> {row.get('status')} ts={row.get('totalsegmentator_status')} slices={row.get('series_slices')}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            row = {
                "case_id": case_id,
                "dicom_folder": str(case_folder),
                "status": "error",
                "error": str(exc),
            }
        rows_by_id[case_id] = row
        processed += 1
        # Incremental flush: a crash / disk-full never loses prior progress.
        df = _write_rows(out_path, rows_by_id, order, base_cols)

    df = _write_rows(out_path, rows_by_id, order, base_cols)
    ok = sum(1 for r in rows_by_id.values() if str(r.get("totalsegmentator_status")) == "ok")
    extracted = sum(1 for r in rows_by_id.values() if r.get("status") == "extracted")
    print("\n" + "=" * 72)
    print(f"JOB DONE -> {out_path}")
    print(f"  rows={len(rows_by_id)} processed={processed} skipped(kept)={skipped}")
    print(f"  status=extracted: {extracted}  totalsegmentator=ok: {ok}")
    if args.canonical:
        kidney_cols = [c for c in df.columns if c.startswith("kidney_") and c.endswith("_volume_cm3")]
        if kidney_cols:
            print(f"  kidney volumes present: {int(df[kidney_cols].notna().any(axis=1).sum())}")


def _limit_threads() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="DICOM prep + extraction pipeline")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Patient folder(s) or archive root(s)",
    )
    parser.add_argument("--dicom-root", type=Path, default=None, help="Alias for a single root folder")
    parser.add_argument("--output", "-o", default="results/extract_from_dicom.csv")
    parser.add_argument(
        "--add-job",
        action="append",
        nargs=2,
        metavar=("ROOT", "OUTPUT"),
        default=[],
        help="Extra (root, output) job; repeatable. E.g. --add-job 'F:/На спине' results/na_spine_full.csv",
    )
    parser.add_argument("--temp-dir", default=None, help="Work dir for NIfTI/segmentation (default: system temp)")
    parser.add_argument("--layout", choices=["auto", "flat", "nested"], default="auto")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--canonical", action="store_true", help="Merge enhanced_ct_extractor body features")
    parser.add_argument("--accuracy-mode", default="balanced", choices=["high", "balanced", "fast", "minimal"])
    parser.add_argument("--no-segmentation", action="store_true", help="Skip TotalSegmentator")
    parser.add_argument("--prep-only", action="store_true", help="Only series selection + dcm2niix")
    parser.add_argument("--nifti-only", action="store_true", help="Same as --prep-only")
    parser.add_argument("--fast", action="store_true", help="Fast TotalSegmentator mode")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--no-compression", action="store_true", help="Uncompressed .nii from dcm2niix")
    parser.add_argument("--reuse-nifti", action="store_true", help="Reuse NIfTI in temp dir")
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Keep good rows in --output and only (re)process failed/new cases",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete per-case scratch dirs (debug; defaults to cleanup)",
    )
    parser.add_argument(
        "--roi-subset",
        nargs="*",
        default=["kidney_right", "kidney_left"],
    )
    args = parser.parse_args()

    _limit_threads()

    jobs: List[Tuple[Path, Path]] = []
    primary_roots = resolve_input_roots(args.paths, args.dicom_root)
    if primary_roots:
        for r in primary_roots:
            jobs.append((r, Path(args.output)))
    for root_str, out_str in args.add_job:
        jobs.append((Path(root_str), Path(out_str)))

    if not jobs:
        print("ERROR: provide at least one path / --dicom-root / --add-job")
        return 2

    work_dir = Path(args.temp_dir) if args.temp_dir else Path(tempfile.gettempdir()) / "ml_trainer_dicom"
    work_dir.mkdir(parents=True, exist_ok=True)
    prep_only = args.prep_only or args.nifti_only

    print("=" * 72)
    print("DICOM PREP + EXTRACTION")
    print("=" * 72)
    print(f"jobs: {[(str(r), str(o)) for r, o in jobs]}")
    print(f"layout: {args.layout}  device: {args.device}  accuracy: {args.accuracy_mode}")
    print(f"dcm2niix: {find_dcm2niix_executable() or 'NOT FOUND'}")
    print(f"TotalSegmentator: {'off' if args.no_segmentation else 'on'}  canonical: {args.canonical}")
    print(f"update_existing: {args.update_existing}  keep_temp: {args.keep_temp}")
    print(f"work_dir: {work_dir}")

    for root, out_path in jobs:
        run_job(root, out_path, args=args, work_dir=work_dir, prep_only=prep_only)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
