"""Mask-status contract for TotalSegmentator kidney ROIs."""

from pathlib import Path

from scripts.inference.extract_from_dicom import inspect_kidney_mask
from src.features.laterality import MASK_STATUS_MISSING_FILE


def test_inspect_kidney_mask_missing_file(tmp_path: Path) -> None:
    status, feats = inspect_kidney_mask(tmp_path / "kidney_left.nii.gz", "kidney_left")
    assert status == MASK_STATUS_MISSING_FILE
    assert feats == {}
