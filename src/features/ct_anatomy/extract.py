"""Assemble one anatomy feature row from masks, or from a segmentation folder."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.features.ct_anatomy.abdomen import assign_clinical_body_size, body_mask_fields, measure_abdomen
from src.features.ct_anatomy.calibration import apply_feature_calibration, load_calibration
from src.features.ct_anatomy.distances import measure_kidney_distances
from src.features.ct_anatomy.kidney_shape import measure_kidney_shape
from src.features.ct_anatomy.perirenal import measure_perirenal
from src.features.ct_anatomy.profiles import MASK_FILES
from src.features.ct_anatomy.psoas import measure_psoas
from src.features.ct_anatomy.qc import QC_MANUAL, body_type_from_bmi, finalize_qc
from src.features.ct_anatomy.spine import measure_spine
from src.features.ct_anatomy.volume import AnatomyVolume

MANUAL_FIELDS = ("diagnosis", "pathology_site", "has_previous_surgery")


def attach_body_type(row: Dict[str, object]) -> Dict[str, object]:
    """Fill ``body_type`` from BMI when the DICOM row does not already have it."""
    if row.get("body_type") is not None:
        return {}
    code = body_type_from_bmi(row.get("bmi"))
    if code is None:
        return {}
    return {
        "body_type": code,
        "body_type_qc": "ok",
        "body_type_source": "bmi_who_excel_scale",
    }


def _manual_placeholders() -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key in MANUAL_FIELDS:
        out[key] = None
        out[f"{key}_qc"] = QC_MANUAL
    return out


def extract_anatomy_features(
    volume: AnatomyVolume,
    *,
    apply_ranges: bool = True,
    calibration: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, object]:
    """Measure every anatomy feature this package knows how to compute."""
    spine = measure_spine(volume)
    features: Dict[str, object] = {}
    features.update(spine)
    features.update(measure_abdomen(volume, _as_float(spine.get("l3_l4_z"))))
    features.update(body_mask_fields(volume, features))
    features.update(assign_clinical_body_size(features))
    features.update(measure_psoas(volume, z_by_side=_kidney_mid_z(volume)))
    features.update(measure_kidney_distances(volume))
    features.update(measure_perirenal(volume))
    features.update(measure_kidney_shape(volume))
    features.update(_manual_placeholders())
    features["anatomy_feature_schema"] = "ct_anatomy_v1"
    if apply_ranges:
        features = finalize_qc(features)
    return apply_feature_calibration(features, calibration if calibration is not None else load_calibration())


def body_mask_fields_from_seg_dir(
    seg_dir: Path,
    hu_nifti: Optional[Path],
    row: Dict[str, object],
) -> Dict[str, object]:
    """Measure skin-to-skin size when the row does not already have one.

    A full-profile row already carries ``body_mask_*`` from L3–L4. The fast
    path only has the body task plus kidney centers, so the slice is the
    kidney level (or the widest skin slice when both kidneys are absent).
    """
    if row.get("body_mask_width_mm") is not None:
        return {}
    folder = Path(seg_dir)
    volume = load_anatomy_volume(folder, Path(hu_nifti) if hu_nifti else None)
    # The body task is 3 mm. A 1.5 mm kidney grid must not drop the skin mask.
    if volume is None or volume.mask("body_trunc") is None:
        volume = _body_only_volume(folder)
    if volume is None or volume.mask("body_trunc") is None:
        return {}
    return body_mask_fields(volume, row)


def _body_only_volume(seg_dir: Path) -> Optional[AnatomyVolume]:
    try:
        import nibabel as nib
    except ImportError:
        return None
    path = next(
        (seg_dir / rel for rel in ("body/body_trunc.nii.gz", "body/body.nii.gz") if (seg_dir / rel).exists()),
        None,
    )
    if path is None:
        return None
    img = nib.load(str(path))
    data = np.asarray(img.dataobj) > 0
    if not np.any(data):
        return None
    return AnatomyVolume(affine=np.asarray(img.affine, dtype=float), masks={"body_trunc": data})


def extract_anatomy_from_seg_dir(
    seg_dir: Path,
    hu_nifti: Optional[Path] = None,
    *,
    calibration: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, object]:
    """Load TotalSegmentator NIfTI masks and measure them. Empty folder → {}."""
    volume = load_anatomy_volume(Path(seg_dir), Path(hu_nifti) if hu_nifti else None)
    if volume is None:
        return {"anatomy_feature_schema": "ct_anatomy_v1", "anatomy_error": "no_masks"}
    return extract_anatomy_features(volume, calibration=calibration)


def load_anatomy_volume(seg_dir: Path, hu_nifti: Optional[Path] = None) -> Optional[AnatomyVolume]:
    try:
        import nibabel as nib
    except ImportError:
        return None

    masks: Dict[str, np.ndarray] = {}
    affine = None
    shape = None
    for name, relatives in MASK_FILES.items():
        path = next((seg_dir / rel for rel in relatives if (seg_dir / rel).exists()), None)
        if path is None:
            continue
        img = nib.load(str(path))
        data = np.asarray(img.dataobj) > 0
        if affine is None:
            affine = np.asarray(img.affine, dtype=float)
            shape = data.shape
        elif data.shape != shape or not np.allclose(np.asarray(img.affine), affine, atol=1e-2):
            continue
        masks[name] = data

    hu = None
    if hu_nifti is not None and hu_nifti.exists():
        hu_img = nib.load(str(hu_nifti))
        hu_data = np.asarray(hu_img.dataobj, dtype=np.float32)
        if affine is None:
            affine = np.asarray(hu_img.affine, dtype=float)
            hu = hu_data
        elif hu_data.shape == shape and np.allclose(np.asarray(hu_img.affine), affine, atol=1e-2):
            hu = hu_data

    if affine is None:
        return None
    return AnatomyVolume(affine=affine, masks=masks, hu=hu)


def _kidney_mid_z(volume: AnatomyVolume) -> Dict[str, Optional[float]]:
    """Cranio-caudal midpoint of each kidney. Missing kidney → no psoas slice."""
    levels: Dict[str, Optional[float]] = {}
    for side in ("left", "right"):
        center = volume.centroid(f"kidney_{side}")
        levels[side] = None if center is None else float(center[2])
    return levels


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number
