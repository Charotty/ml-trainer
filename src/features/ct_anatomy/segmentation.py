"""Run TotalSegmentator tasks for the anatomy profile and reuse cached masks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from src.features.ct_anatomy.profiles import FULL_TASKS, PROFILE_FAST, PROFILE_FULL

Runner = Callable[..., bool]


def _has_mask_files(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    return any(folder.glob("*.nii*"))


def _resolve_device(device: str) -> str:
    try:
        from scripts.inference.dicom_prep import resolve_totalsegmentator_device

        return resolve_totalsegmentator_device(device)
    except Exception:
        return "cpu" if device == "auto" else device


def default_totalsegmentator_runner(
    *,
    input_path: Path,
    output: Path,
    task: str,
    roi_subset: Optional[Sequence[str]],
    fast: bool,
    device: str,
) -> bool:
    """Call TotalSegmentator. Retries once in ``--fast`` after an out-of-memory error."""
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError:
        print("  TotalSegmentator not installed (pip install TotalSegmentator)")
        return False

    output.mkdir(parents=True, exist_ok=True)
    dev = _resolve_device(device)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    kwargs: Dict[str, Any] = {
        "input": str(input_path),
        "output": str(output),
        "task": task,
        "nr_thr_resamp": 1,
        "nr_thr_saving": 1,
        "fast": fast,
        "quiet": True,
        "device": dev,
    }
    if roi_subset:
        kwargs["roi_subset"] = list(roi_subset)

    def _call(options: Dict[str, Any]) -> None:
        try:
            totalsegmentator(**options)
        except SystemExit as exc:
            # Licensed tasks such as vertebrae_body call sys.exit instead of raising.
            raise RuntimeError(f"TotalSegmentator exited ({exc.code})") from exc
        except TypeError:
            fallback = dict(options)
            fallback.pop("roi_subset", None)
            if task == "total":
                fallback.pop("task", None)
            try:
                totalsegmentator(**fallback)
            except SystemExit as exc:
                raise RuntimeError(f"TotalSegmentator exited ({exc.code})") from exc

    try:
        print(f"  TotalSegmentator task={task} fast={fast} device={dev}")
        _call(kwargs)
    except MemoryError as exc:
        if fast:
            print(f"  TotalSegmentator memory error ({task}): {exc}")
            return False
        print(f"  memory error on {task}, retrying 3 mm model: {exc}")
        kwargs["fast"] = True
        try:
            _call(kwargs)
        except Exception as retry_exc:
            print(f"  TotalSegmentator error ({task}): {retry_exc}")
            return False
        return _has_mask_files(output)
    except Exception as exc:
        print(f"  TotalSegmentator error ({task}): {exc}")
        return False
    return _has_mask_files(output)


def run_body_task(
    nifti_path: Path,
    seg_dir: Path,
    *,
    device: str = "auto",
    reuse: bool = True,
    fast: bool = True,
    runner: Optional[Runner] = None,
) -> Dict[str, Any]:
    """Segment the skin envelope into ``seg_dir/body``.

    The 3 mm model is enough for width and depth. A cached ``*.nii*`` in that
    folder is kept when ``reuse`` is set.
    """
    seg_dir = Path(seg_dir)
    out = seg_dir / "body"
    if reuse and _has_mask_files(out):
        return {"anatomy_seg_status_body": "cached"}
    call = runner or default_totalsegmentator_runner
    ok = call(
        input_path=Path(nifti_path),
        output=out,
        task="body",
        roi_subset=None,
        fast=fast,
        device=device,
    )
    return {"anatomy_seg_status_body": "ok" if ok else "failed"}


def run_anatomy_segmentation(
    nifti_path: Path,
    seg_dir: Path,
    profile: str = PROFILE_FAST,
    *,
    device: str = "auto",
    reuse: bool = True,
    runner: Optional[Runner] = None,
) -> Dict[str, Any]:
    """Segment ``nifti_path`` into ``seg_dir/<task>/``.

    ``profile='fast'`` does nothing here: the kidney-only call in
    ``extract_from_dicom`` stays responsible for that path. Cached task
    folders (any ``*.nii*``) are kept when ``reuse`` is set.
    """
    row: Dict[str, Any] = {"anatomy_profile": profile}
    if profile != PROFILE_FULL:
        return row

    seg_dir = Path(seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
    call = runner or default_totalsegmentator_runner
    nifti_path = Path(nifti_path)

    # vertebrae_body cannot use --fast and its 1.5 mm model is enough to trip
    # the WSL GPU driver. The 3 mm tasks stay on the requested device.
    cpu_tasks = {"vertebrae_body"}
    for task, roi_subset, fast in FULL_TASKS:
        out = seg_dir / task
        status_key = f"anatomy_seg_status_{task}"
        if reuse and _has_mask_files(out):
            row[status_key] = "cached"
            continue
        task_device = "cpu" if task in cpu_tasks and device != "cpu" else device
        ok = call(
            input_path=nifti_path,
            output=out,
            task=task,
            roi_subset=roi_subset,
            fast=fast,
            device=task_device,
        )
        row[status_key] = "ok" if ok else "failed"
    return row
