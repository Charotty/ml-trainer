"""Production adaptive-ensemble package (Block 6).

Status: production. Legacy entrypoint: ``models/phase1/adaptive_ensemble.py``
(thin re-export). Experimental trainers remain under ``src/models/train_*.py``.
"""

from __future__ import annotations

PACKAGE_STATUS = "production"

from src.models.ensemble.estimators import (  # noqa: E402
    MODEL_KIND_ALIASES,
    make_base_models,
)
from src.models.ensemble.features import (  # noqa: E402
    FeatureTransformer,
    inference_transformer_from_payload,
)
from src.models.ensemble.trainer import AdaptiveEnsembleTrainer  # noqa: E402

__all__ = [
    "PACKAGE_STATUS",
    "AdaptiveEnsembleTrainer",
    "FeatureTransformer",
    "MODEL_KIND_ALIASES",
    "inference_transformer_from_payload",
    "make_base_models",
]
