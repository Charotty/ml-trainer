"""Extractor must not emit hardcoded displacement targets."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "scripts" / "inference" / "enhanced_ct_extractor.py"


def test_extractor_no_hardcoded_delta_literals():
    source = EXTRACTOR.read_text(encoding="utf-8")
    bad = ("12.5", "4.2", "8.1", "-8.3", "3.8", "7.9")
    for literal in bad:
        assert f"_delta_x'] = {literal}" not in source
        assert f"_delta_y'] = {literal}" not in source
        assert f"_delta_z'] = {literal}" not in source
    assert "float('nan')" in source or 'float("nan")' in source


def test_extractor_uses_ct_geometry_module():
    text = EXTRACTOR.read_text(encoding="utf-8")
    assert "ct_geometry" in text
    assert "merge_spine_relative" in text
