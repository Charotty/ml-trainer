#!/usr/bin/env python3
"""
DICOM preparation utilities for amImageViewer / PACS export layouts.

Handles:
- extensionless DICOM files (DICM header)
- mixed series in one patient folder (root cause of TotalSegmentator MISSING_DICOM_FILES)
- flat patient roots (e.g. F:/На Боку) and nested archives (e.g. F:/На спине)
- dcm2niix binary discovery and primary 3D NIfTI selection
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None  # type: ignore[assignment]

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None  # type: ignore[assignment]


_SKIP_SERIES_RE = re.compile(
    r"localizer|scout|topo|survey|mparts|dose|report|screen\s*save|smart\s*prep",
    re.IGNORECASE,
)

_NON_DICOM_SUFFIXES = {
    ".exe", ".inf", ".cds", ".txt", ".html", ".xml", ".pdf", ".zip",
    ".png", ".jpg", ".svg", ".qm", ".mng", ".dll", ".bat", ".ini",
}


@dataclass
class DicomSeriesInfo:
    series_uid: str
    files: List[Path] = field(default_factory=list)
    modality: str = ""
    description: str = ""
    series_number: Optional[int] = None
    rows: Optional[int] = None
    columns: Optional[int] = None

    @property
    def slice_count(self) -> int:
        return len(self.files)


@dataclass
class PrepResult:
    case_id: str
    work_slug: str
    source_folder: Path
    series: DicomSeriesInfo
    series_input_dir: Path
    nifti_file: Optional[Path] = None
    nifti_files: List[Path] = field(default_factory=list)
    staging_dir: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)


def make_ascii_work_slug(label: str, *, index: Optional[int] = None) -> str:
    """
    ASCII-only directory token for temp paths.

    TotalSegmentator/nibabel fail on Cyrillic or spaces in output paths
    (e.g. seg_Абдурахманов М.А. - 05.11.2025).
    """
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:12]
    if index is not None:
        return f"case_{index:04d}_{digest}"
    return f"case_{digest}"


def is_dicom_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def iter_dicom_paths(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in _NON_DICOM_SUFFIXES:
            continue
        if path.suffix.lower() in {"", ".dcm"} and is_dicom_file(path):
            yield path


def count_dicom_files(folder: Path) -> int:
    return sum(1 for _ in iter_dicom_paths(folder))


def count_dicom_files_in_dir_direct(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for f in directory.iterdir() if f.is_file() and is_dicom_file(f))


def count_candidate_slices_subtree(folder: Path) -> int:
    """Name-based candidate-slice total over the whole subtree (no file opens)."""
    total = 0
    stack = [str(folder)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif _is_candidate_slice_name(entry.name) and entry.is_file(follow_symlinks=False):
                            total += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def find_dominant_dicom_subdir(folder: Path) -> Optional[Path]:
    """
    Find the subdirectory that contains the bulk of DICOM slices.

    amImageViewer exports (На Боку) store ~4k slices in a single numeric folder
    (e.g. 25111414/). Name-based counting (no header opens) keeps this cheap on
    the Windows DrvFs mount; header parsing happens later only on the winner.
    """
    best_n = 0
    best_dir: Optional[Path] = None
    total = 0
    stack = [str(folder)]
    while stack:
        current = stack.pop()
        direct = 0
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif _is_candidate_slice_name(entry.name) and entry.is_file(follow_symlinks=False):
                            direct += 1
                    except OSError:
                        continue
        except OSError:
            continue
        total += direct
        if direct > best_n:
            best_n = direct
            best_dir = Path(current)

    if total == 0:
        return None
    if best_dir and best_n >= max(200, int(total * 0.75)):
        return best_dir
    return None


def _read_series_header(path: Path) -> Optional[Tuple[str, str, str, Optional[int], Optional[int], Optional[int]]]:
    if pydicom is None:
        raise RuntimeError("pydicom is required for DICOM preparation")
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        suid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
        modality = str(getattr(ds, "Modality", "") or "").upper()
        description = str(getattr(ds, "SeriesDescription", "") or "")
        series_number = getattr(ds, "SeriesNumber", None)
        rows = getattr(ds, "Rows", None)
        columns = getattr(ds, "Columns", None)
        try:
            series_number = int(series_number) if series_number is not None else None
        except (TypeError, ValueError):
            series_number = None
        return suid, modality, description, series_number, rows, columns
    except Exception:
        return None


def iter_candidate_paths(folder: Path) -> Iterable[Path]:
    """Name-based candidate slice paths (no DICM-magic open; dcmread validates)."""
    stack = [str(folder)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif _is_candidate_slice_name(entry.name) and entry.is_file(follow_symlinks=False):
                            yield Path(entry.path)
                    except OSError:
                        continue
        except OSError:
            continue


def group_dicom_series(folder: Path, max_files: Optional[int] = None) -> Dict[str, DicomSeriesInfo]:
    if pydicom is None:
        raise RuntimeError("pydicom is required for DICOM preparation")

    series_map: Dict[str, DicomSeriesInfo] = {}
    for i, fp in enumerate(iter_candidate_paths(folder)):
        if max_files is not None and i >= max_files:
            break
        header = _read_series_header(fp)
        if header is None:
            continue
        suid, modality, description, series_number, rows, columns = header
        info = series_map.get(suid)
        if info is None:
            info = DicomSeriesInfo(
                series_uid=suid,
                modality=modality,
                description=description,
                series_number=series_number,
                rows=rows,
                columns=columns,
            )
            series_map[suid] = info
        info.files.append(fp)
    return series_map


def _series_score(info: DicomSeriesInfo) -> Tuple[int, int, int]:
    desc_penalty = 1 if _SKIP_SERIES_RE.search(info.description or "") else 0
    modality_penalty = 0 if info.modality == "CT" else 10
    size_bonus = info.slice_count
    dim_bonus = 0
    if info.rows and info.columns:
        if info.rows >= 256 and info.columns >= 256:
            dim_bonus = 1000
        elif info.rows >= 128 and info.columns >= 128:
            dim_bonus = 100
    return (modality_penalty + desc_penalty, -size_bonus - dim_bonus, info.series_number or 9999)


def select_main_ct_series(series_map: Dict[str, DicomSeriesInfo]) -> Optional[DicomSeriesInfo]:
    if not series_map:
        return None

    ct_series = [s for s in series_map.values() if s.modality == "CT" and s.slice_count >= 50]
    candidates = ct_series or list(series_map.values())
    candidates = [s for s in candidates if not _SKIP_SERIES_RE.search(s.description or "")]
    if not candidates:
        candidates = list(series_map.values())

    return min(candidates, key=_series_score)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return
    except OSError:
        shutil.copy2(src, dst)


def build_series_input_dir(
    series: DicomSeriesInfo,
    work_dir: Path,
    work_slug: str,
) -> Tuple[Path, Optional[Path]]:
    """Return a directory containing only the chosen series (staging if needed)."""
    if not series.files:
        raise ValueError("empty DICOM series")

    parents = {f.parent for f in series.files}
    if len(parents) == 1:
        only_parent = next(iter(parents))
        # If the chosen series is the only thing in its directory, point dcm2niix
        # straight at it (no staging copy). Name-based count avoids file opens.
        if count_candidate_slices_direct(only_parent) == len(series.files):
            return only_parent, None

    staging = work_dir / f"series_{work_slug}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(sorted(series.files, key=lambda p: p.name)):
        dst = staging / f"{i:06d}{src.suffix if src.suffix else '.dcm'}"
        _link_or_copy(src, dst)

    return staging, staging


def _dcm2niix_install_hint() -> str:
    if sys.platform == "win32":
        return "pip install dcm2niix (or add dcm2niix.exe to PATH)"
    if sys.platform == "darwin":
        return "brew install dcm2niix or pip install dcm2niix"
    return "apt install dcm2niix or pip install dcm2niix"


def find_dcm2niix_executable() -> Optional[str]:
    cmd = shutil.which("dcm2niix")
    if cmd:
        return cmd

    exe_parent = Path(sys.executable).resolve().parent
    search_dirs = [exe_parent, exe_parent / "Scripts", exe_parent / "bin"]
    for directory in search_dirs:
        for name in ("dcm2niix.exe", "dcm2niix"):
            candidate = directory / name
            if candidate.exists():
                return str(candidate)

    try:
        import dcm2niix as _dcm2niix_pkg

        pkg_exe = Path(_dcm2niix_pkg.__file__).resolve().parent / "dcm2niix.exe"
        if pkg_exe.exists():
            return str(pkg_exe)
        pkg_bin = pkg_exe.with_suffix("")
        if pkg_bin.exists():
            return str(pkg_bin)
    except ImportError:
        pass

    return None


def convert_dicom_to_nifti(
    dicom_folder: Path,
    output_folder: Path,
    *,
    compress: bool = True,
    filename_pattern: str = "%p_%s",
    timeout_seconds: int = 1800,
) -> Tuple[bool, List[Path], str]:
    dcm2niix_cmd = find_dcm2niix_executable()
    if not dcm2niix_cmd:
        return False, [], f"dcm2niix binary not found (install: {_dcm2niix_install_hint()})"

    output_folder.mkdir(parents=True, exist_ok=True)
    cmd = [
        dcm2niix_cmd,
        "-z", "y" if compress else "n",
        "-f", filename_pattern,
        "-o", str(output_folder),
        "-b", "n",
        "-m", "y",
        "-x", "n",
        str(dicom_folder),
    ]
    last_err = ""
    for attempt in range(2):
        if attempt:
            time.sleep(2)
            shutil.rmtree(output_folder, ignore_errors=True)
            output_folder.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            return False, [], f"dcm2niix timeout ({timeout_seconds}s)"
        except Exception as exc:
            return False, [], str(exc)

        nifti_files = sorted(output_folder.glob("*.nii*"))
        if nifti_files:
            return True, [str(p) for p in nifti_files], ""

        last_err = (result.stderr or result.stdout or "dcm2niix failed").strip()
        if result.returncode == 0:
            return False, [], "dcm2niix produced no NIfTI files"

    return False, [], last_err


def pick_primary_ct_nifti(nifti_files: Sequence[Path | str]) -> Optional[Path]:
    if not nifti_files:
        return None
    if nib is None:
        return Path(str(nifti_files[0]))

    best: Optional[Path] = None
    best_score = (-1, -1.0)
    for item in nifti_files:
        path = Path(item)
        try:
            img = nib.load(str(path))
            shape = img.shape
            if len(shape) < 3:
                continue
            z = int(shape[2]) if len(shape) == 3 else int(shape[3])
            vox = float(np.prod(img.header.get_zooms()[:3])) if hasattr(img, "header") else 1.0
            score = (z, path.stat().st_size)
            if score > best_score:
                best_score = score
                best = path
        except Exception:
            continue

    if best is not None:
        return best
    try:
        return Path(str(max(nifti_files, key=lambda p: Path(p).stat().st_size)))
    except Exception:
        return Path(str(nifti_files[0]))


def has_dicom_content(folder: Path) -> bool:
    """Fast check whether a folder likely contains a CT series."""
    if (folder / "DICOMDIR").exists():
        return True
    for child in folder.iterdir():
        if child.is_file() and is_dicom_file(child):
            return True
        if child.is_dir() and count_dicom_files_in_dir_direct(child) >= 50:
            return True
    return False


# Minimum slices in a single directory for it to count as a CT series subtree.
_MIN_SERIES_SLICES = 150


def _is_candidate_slice_name(name: str) -> bool:
    """
    Name-only DICOM-slice heuristic (NO file open).

    Discovery walks every patient up front; opening each extensionless file to
    confirm the DICM magic is prohibitively slow on the Windows DrvFs mount.
    Series-export slices are extensionless or ``.dcm``; the real header check
    still happens later in ``group_dicom_series`` / ``prepare_case``.
    """
    dot = name.rfind(".")
    if dot < 0:
        return True  # extensionless (typical amImageViewer / PACS slice)
    return name[dot:].lower() == ".dcm"


def count_candidate_slices_direct(directory: Path) -> int:
    count = 0
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_file() and _is_candidate_slice_name(entry.name):
                    count += 1
    except OSError:
        return 0
    return count


def max_series_size_in_subtree(folder: Path, min_hit: int = _MIN_SERIES_SLICES) -> int:
    """
    Largest candidate-slice count in any single directory of the subtree.

    Filesystem-listing only (no file opens) with an explicit scandir stack so
    that a directory stops being enumerated the instant it reaches ``min_hit``
    (image dirs hold thousands of slices; on DrvFs every entry is expensive).
    """
    best = 0
    stack = [str(folder)]
    while stack:
        current = stack.pop()
        count = 0
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif _is_candidate_slice_name(entry.name) and entry.is_file(follow_symlinks=False):
                            count += 1
                            if count >= min_hit:
                                return min_hit
                    except OSError:
                        continue
        except OSError:
            continue
        if count > best:
            best = count
    return best


def has_ct_series_subtree(folder: Path, min_slices: int = _MIN_SERIES_SLICES) -> bool:
    return max_series_size_in_subtree(folder, min_hit=min_slices) >= min_slices


def _dir_has_direct_series(folder: Path, min_slices: int = _MIN_SERIES_SLICES) -> bool:
    """True when this directory itself is the image folder (slices live here)."""
    return count_candidate_slices_direct(folder) >= min_slices


def discover_patient_cases(root: Path, layout: str = "auto", max_unwrap_depth: int = 6) -> List[Path]:
    """
    Discover one folder per patient case under a DICOM root.

    Handles two real-world layouts:
      * flat (F:/На Боку): root -> 109 patient folders -> numeric image dir
      * nested (F:/На спине): root -> single archive wrapper -> per-patient
        folders -> (numeric dir | PACS PA*/ST*/SE* tree)

    ``layout='flat'`` forces the direct children of ``root``. ``auto`` / ``nested``
    unwrap single grouping wrappers until the level with >= 2 sibling patient
    folders is reached.
    """
    root = Path(root)
    if not root.exists():
        return []
    if not root.is_dir():
        return [root]

    if layout == "flat":
        return sorted(c for c in root.iterdir() if c.is_dir() and has_ct_series_subtree(c))

    node = root
    for _ in range(max_unwrap_depth):
        subdirs = [c for c in node.iterdir() if c.is_dir()]
        case_children = [c for c in subdirs if has_ct_series_subtree(c)]

        if len(case_children) >= 2:
            return sorted(case_children)
        if len(case_children) == 1 and not _dir_has_direct_series(node):
            node = case_children[0]
            continue
        break

    if has_ct_series_subtree(node) or _dir_has_direct_series(node):
        return [node]
    return []


def cleanup_case_temp(work_dir: Path, work_slug: str) -> None:
    """Remove all per-case scratch directories for a slug (tmpfs/disk hygiene)."""
    for prefix in ("series_", "nifti_", "seg_"):
        target = work_dir / f"{prefix}{work_slug}"
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)


def prepare_case(
    case_folder: Path,
    work_dir: Path,
    *,
    case_id: Optional[str] = None,
    work_slug: Optional[str] = None,
    case_index: Optional[int] = None,
    compress_nifti: bool = True,
    reuse_nifti: bool = False,
) -> PrepResult:
    """Select main CT series, stage if needed, convert to NIfTI."""
    case_folder = Path(case_folder)
    cid = case_id or case_folder.name
    slug = work_slug or make_ascii_work_slug(cid, index=case_index)
    warnings: List[str] = []

    scan_root = find_dominant_dicom_subdir(case_folder) or case_folder
    if scan_root != case_folder:
        warnings.append(f"dominant_subdir={scan_root.name}")

    series_map = group_dicom_series(scan_root)
    if not series_map:
        raise ValueError(f"No DICOM series found in {case_folder}")

    series = select_main_ct_series(series_map)
    total_dicoms = count_candidate_slices_subtree(case_folder)
    needs_rescan = (
        series is not None
        and total_dicoms > series.slice_count * 2
        and series.slice_count < 500
    )
    if needs_rescan and scan_root != case_folder:
        warnings.append("rescanned_full_patient_tree")
        series_map = group_dicom_series(case_folder)
        series = select_main_ct_series(series_map)

    if not series_map:
        raise ValueError(f"No DICOM series found in {case_folder}")

    if len(series_map) > 1:
        warnings.append(f"multiple_series={len(series_map)}")

    if series is None:
        series = select_main_ct_series(series_map)
    if series is None:
        raise ValueError(f"No suitable CT series in {case_folder}")

    if series.modality != "CT":
        warnings.append(f"selected_modality={series.modality}")

    series_input_dir, staging_dir = build_series_input_dir(series, work_dir, slug)

    nifti_dir = work_dir / f"nifti_{slug}"
    nifti_file: Optional[Path] = None
    nifti_files: List[Path] = []

    existing = sorted(nifti_dir.glob("*.nii*")) if nifti_dir.exists() else []
    if reuse_nifti and existing:
        nifti_files = existing
        nifti_file = pick_primary_ct_nifti(existing)
    else:
        if nifti_dir.exists():
            shutil.rmtree(nifti_dir, ignore_errors=True)
        ok, files, err = convert_dicom_to_nifti(series_input_dir, nifti_dir, compress=compress_nifti)
        if not ok:
            raise RuntimeError(f"dcm2niix failed: {err}")
        nifti_files = [Path(f) for f in files]
        nifti_file = pick_primary_ct_nifti(nifti_files)

    return PrepResult(
        case_id=cid,
        work_slug=slug,
        source_folder=case_folder,
        series=series,
        series_input_dir=series_input_dir,
        nifti_file=nifti_file,
        nifti_files=nifti_files,
        staging_dir=staging_dir,
        warnings=warnings,
    )


def resolve_totalsegmentator_device(requested: str = "auto") -> str:
    req = (requested or "auto").lower()
    if req in {"cpu", "gpu"}:
        return req
    try:
        import torch

        return "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
