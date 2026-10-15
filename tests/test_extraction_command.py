"""Workbench extraction always asks for the vertebral frame."""

from pathlib import Path

from src.api.cases.worker import build_extraction_command


def test_fast_analyze_still_requests_vertebral_frame(tmp_path: Path):
    cmd = build_extraction_command(
        dicom_dir=tmp_path / "dicom",
        work_dir=tmp_path / "work",
        case_id="case-1",
        result_path=tmp_path / "out.json",
        fast=True,
    )
    assert "--fast" in cmd
    assert "--vertebral-frame" in cmd
    assert cmd.index("--vertebral-frame") < cmd.index("--fast")
