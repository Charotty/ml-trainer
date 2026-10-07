"""Outer-CV evaluator lives in ``src.models.nested_cv`` (Block 2)."""

from __future__ import annotations

PACKAGE_STATUS = "production"

from src.models.nested_cv import (  # noqa: E402
    evaluate_nested_groupkfold_oof,
    evaluate_nested_proxy_vs_honest,
)

__all__ = [
    "PACKAGE_STATUS",
    "evaluate_nested_groupkfold_oof",
    "evaluate_nested_proxy_vs_honest",
]
