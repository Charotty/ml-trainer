"""Shared deprecation helper for non-canonical feature / train scripts."""

from __future__ import annotations

import warnings

CANONICAL_HINT = (
    "Canonical Phase 1 path: scripts/run_phase1_pipeline.py info "
    "(schema: src/features/phase1_schema.py)"
)


def warn_legacy_script(module_name: str, *, replacement: str) -> None:
    warnings.warn(
        f"{module_name} is legacy. Use {replacement} instead. {CANONICAL_HINT}",
        DeprecationWarning,
        stacklevel=3,
    )
