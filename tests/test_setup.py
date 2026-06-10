"""Smoke-import tests for critical project modules (audit fix 1.12)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CRITICAL_MODULES = [
    "features.phase1_schema",
    "ar_system.kidney_ar_system",
    "reliability.confidence_constraints",
    "data_validation",
    "validation.data_validator",
    "versioning.version_manager",
]


@pytest.mark.parametrize("module_path", CRITICAL_MODULES)
def test_critical_module_imports(module_path: str) -> None:
    """Each listed module must import without raising."""
    module = importlib.import_module(module_path)
    assert module is not None
