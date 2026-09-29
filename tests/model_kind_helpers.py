"""Shared smoke test for stage-7 model_kind plugins (synthetic nested CV)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.phase1_schema import BASE_FEATURES, TARGET_NAMES
from src.models.baselines import normalize_model_kind
from src.models.ensemble import AdaptiveEnsembleTrainer, model_kinds
from src.models.nested_cv import evaluate_nested_groupkfold_oof


def _synthetic(n: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = {c: float(100 + rng.normal(0, 5)) for c in BASE_FEATURES}
        row["full_name"] = f"P{i:02d}"
        signal = (row[BASE_FEATURES[0]] - 100) * 0.8
        for t in TARGET_NAMES:
            row[t] = float(signal + rng.normal(0, 1))
        rows.append(row)
    return pd.DataFrame(rows)


def run_kind(kind: str) -> float:
    model_kinds.load_all()
    assert model_kinds.is_registered(kind)
    assert normalize_model_kind(kind) == model_kinds.resolve(kind)
    df = _synthetic()

    def factory(**kw):
        return AdaptiveEnsembleTrainer(
            enrichment_mode="none",
            estimator_profile="tiny",
            inner_n_splits=2,
            model_kind=kind,
            **kw,
        )

    result = evaluate_nested_groupkfold_oof(
        df, trainer_factory=factory, n_splits=3, group_seed=0, targets=list(TARGET_NAMES)
    )
    mae = float(result.metrics["avg_mae_mm"])
    assert np.isfinite(mae)
    return mae


def median_mae() -> float:
    return run_kind_builtin("median")


def run_kind_builtin(kind: str) -> float:
    df = _synthetic()

    def factory(**kw):
        return AdaptiveEnsembleTrainer(
            enrichment_mode="none", estimator_profile="tiny", inner_n_splits=2, model_kind=kind, **kw
        )

    result = evaluate_nested_groupkfold_oof(
        df, trainer_factory=factory, n_splits=3, group_seed=0, targets=list(TARGET_NAMES)
    )
    return float(result.metrics["avg_mae_mm"])


__all__ = ["run_kind", "median_mae", "pytest"]
