"""Extractor must not emit hardcoded/synthetic displacement targets; prefer determinism."""

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


def test_extractor_deltas_are_nan_not_synthetic():
    text = EXTRACTOR.read_text(encoding="utf-8")
    assert 'out[f"{prefix}_delta_x"] = float("nan")' in text
    assert 'out[f"{prefix}_delta_y"] = float("nan")' in text
    assert 'out[f"{prefix}_delta_z"] = float("nan")' in text
    assert "no synthetic deltas" in text


def test_extractor_no_random_sampling_apis():
    text = EXTRACTOR.read_text(encoding="utf-8")
    forbidden = (
        "random.shuffle",
        "random.choice",
        "np.random.choice",
        "np.random.shuffle",
        "np.random.randint",
        "np.random.rand(",
        "np.random.randn(",
    )
    for token in forbidden:
        assert token not in text, f"unexpected non-deterministic call: {token}"
    assert "np.linspace" in text
    assert "--seed" in text
    assert "_set_extractor_seed" in text
