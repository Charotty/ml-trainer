#!/usr/bin/env python3
"""Legacy shim. Production trainer: ``src.models.ensemble.AdaptiveEnsembleTrainer``."""

from __future__ import annotations

import warnings

PACKAGE_STATUS = "legacy"

warnings.warn(
    "models.phase1.adaptive_ensemble is a legacy re-export. "
    "Import AdaptiveEnsembleTrainer from src.models.ensemble.",
    DeprecationWarning,
    stacklevel=2,
)

from src.models.ensemble.estimators import (  # noqa: E402,F401
    MODEL_KIND_ALIASES,
    SINGLE_KIND_TO_NAME,
)
from src.models.ensemble.trainer import AdaptiveEnsembleTrainer  # noqa: E402

__all__ = ["AdaptiveEnsembleTrainer", "MODEL_KIND_ALIASES", "PACKAGE_STATUS"]


def main():
    from src.models.ensemble.trainer import main as _main

    return _main()


if __name__ == "__main__":
    main()
