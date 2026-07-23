"""Tests for API model_info honesty and legacy --force-legacy gate (stages 5–6)."""

from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "src" / "api") not in sys.path:
    sys.path.insert(0, str(ROOT / "src" / "api"))


def test_performance_from_training_meta_unavailable_without_metrics() -> None:
    from kidney_displacement_api import _performance_from_training_meta

    perf = _performance_from_training_meta({"training_meta": {"clinical_only": True}})
    assert perf["status"] == "unavailable"
    assert perf["average_mae_mm"] is None


def test_performance_from_training_meta_uses_embedded_metrics() -> None:
    from kidney_displacement_api import _performance_from_training_meta

    perf = _performance_from_training_meta(
        {
            "training_meta": {
                "performance": {
                    "average_mae_mm": 3.5,
                    "average_r2": 0.2,
                    "within_5mm_ratio": 0.8,
                    "within_10mm_ratio": 0.95,
                }
            }
        }
    )
    assert perf["status"] == "from_training_meta"
    assert perf["average_mae_mm"] == 3.5
    assert perf["accuracy_5mm"] == 0.8


def test_performance_missing_training_meta() -> None:
    from kidney_displacement_api import _performance_from_training_meta

    perf = _performance_from_training_meta({"models": {}})
    assert perf["status"] == "unavailable"
    assert "missing" in perf["detail"]


def test_legacy_force_gate_exits_without_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.features.legacy_force_gate import warn_and_require_force_legacy

    monkeypatch.setattr(sys, "argv", ["train_lasso.py"])
    with pytest.raises(SystemExit) as exc:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warn_and_require_force_legacy("models/phase1/train_lasso.py")
    assert "DEPRECATED" in str(exc.value)


def test_legacy_force_gate_allows_with_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.features.legacy_force_gate import warn_and_require_force_legacy

    monkeypatch.setattr(sys, "argv", ["train_lasso.py", "--force-legacy", "--other"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        warn_and_require_force_legacy("models/phase1/train_lasso.py")
    assert "--force-legacy" not in sys.argv
    assert "--other" in sys.argv
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_cors_origins_not_star_with_credentials() -> None:
    path = ROOT / "src" / "api" / "api_server.py"
    source = path.read_text(encoding="utf-8")
    assert 'allow_origins=["*"]' not in source
    assert "allow_credentials=True" in source
    assert "CORS_ALLOW_ORIGINS" in source


def test_legacy_train_lasso_main_exits_without_force() -> None:
    script = ROOT / "models" / "phase1" / "train_lasso.py"
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script), run_name="__main__")
    assert exc.value.code != 0
    assert "DEPRECATED" in str(exc.value)
