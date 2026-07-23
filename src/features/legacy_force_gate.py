"""Helpers for deprecated Phase-1 CLI entrypoints."""

from __future__ import annotations

import sys
import warnings

CANONICAL_TRAIN = "scripts/data/train_clinical_honest.py"
CANONICAL_API = "src/api/kidney_displacement_api.py"


def warn_and_require_force_legacy(
    script_name: str,
    *,
    replacement: str = CANONICAL_TRAIN,
) -> None:
    """Emit DeprecationWarning and exit unless ``--force-legacy`` is passed.

    Strips ``--force-legacy`` from ``sys.argv`` when present so downstream
    parsers / ``main()`` see a clean argv.
    """
    force = "--force-legacy" in sys.argv
    warnings.warn(
        f"{script_name} is deprecated. Use {replacement} instead. "
        "Pass --force-legacy to run this legacy entrypoint anyway.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not force:
        raise SystemExit(
            f"[DEPRECATED] {script_name}\n"
            f"Canonical path: {replacement}\n"
            "Re-run with --force-legacy only if you intentionally need the legacy script."
        )
    sys.argv = [a for a in sys.argv if a != "--force-legacy"]
