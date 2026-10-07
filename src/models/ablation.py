"""Feature-group ablation experiment API (Block 3.4).

Runs nested GroupKFold with groups dropped. Does **not** remove groups from
production on a single run — callers must keep the production feature set
until paired repeated-CV justifies a change.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from src.features.displacement_axis_features import ANATOMICAL_FEATURES, DISPLACEMENT_AXIS_FEATURES
from src.features.phase1_schema import (
    BASE_FEATURES,
    CLINICAL_DEMOGRAPHIC_FEATURES,
    CROSS_FEATURES,
    ENGINEERED_FEATURES,
    TARGET_NAMES,
)
from src.models.baselines import make_trainer_factory
from src.models.nested_cv import NestedOOFResult, evaluate_nested_groupkfold_oof

# Sequential groups from the plan. Prefix groups are tested separately.
ABLATION_GROUPS: dict[str, tuple[str, ...]] = {
    "base": tuple(BASE_FEATURES),
    "demographics": tuple(CLINICAL_DEMOGRAPHIC_FEATURES),
    "engineered": tuple(list(ENGINEERED_FEATURES) + list(CROSS_FEATURES) + list(DISPLACEMENT_AXIS_FEATURES)),
    "anatomical_extras": tuple(ANATOMICAL_FEATURES),
}

NA_POP_SHIFT_PREFIXES: tuple[str, ...] = ("na_pop_shift_",)
NA_SUP_AFFINE_PREFIXES: tuple[str, ...] = ("na_sup_z_", "na_sup_pct_")
NA_TREND_PREFIXES: tuple[str, ...] = (
    "na_pop_shift_",
    "na_sup_z_",
    "na_sup_pct_",
    "kits_z_",
    "kits_pct_",
    "kits_cohort_median_",
)

CUMULATIVE_ORDER: tuple[str, ...] = (
    "base",
    "demographics",
    "engineered",
    "anatomical_extras",
    "na_trends",
)

PRODUCTION_DROP_FORBIDDEN_NOTE = (
    "Ablation results are experimental. Do not drop a feature group from "
    "production on a single run; require paired repeated nested CV first."
)


def group_drop_spec(group: str) -> dict[str, Any]:
    """Translate a named group into trainer drop arguments."""
    key = str(group)
    if key in ABLATION_GROUPS:
        return {"drop_feature_groups": (key,)}
    if key == "na_trends":
        return {"drop_feature_prefixes": NA_TREND_PREFIXES}
    if key in {"na_pop_shift", "na_pop_shift_*"}:
        return {"drop_feature_prefixes": NA_POP_SHIFT_PREFIXES}
    if key in {"na_sup", "na_sup_*", "na_sup_affine"}:
        return {"drop_feature_prefixes": NA_SUP_AFFINE_PREFIXES}
    raise ValueError(f"Unknown ablation group {group!r}")


def evaluate_ablation_nested(
    df: pd.DataFrame,
    *,
    n_splits: int = 2,
    model_kind: str = "ensemble",
    groups: Sequence[str] | None = None,
    include_full: bool = True,
    trainer_kwargs: Mapping[str, Any] | None = None,
    targets: list[str] | None = None,
    yz_target_boost: bool = False,
) -> dict[str, Any]:
    """Leave-one-group-out nested OOF. Never mutates a production artifact."""
    cols = list(targets or [c for c in TARGET_NAMES if c in df.columns])
    kwargs = dict(trainer_kwargs or {})
    kwargs.setdefault("enrichment_mode", "none")
    kwargs.setdefault("estimator_profile", "tiny")
    kwargs.setdefault("inner_n_splits", 2)
    kwargs["yz_target_boost"] = bool(yz_target_boost)

    names = list(groups or (CUMULATIVE_ORDER + ("na_pop_shift", "na_sup_affine")))
    runs: dict[str, Any] = {}

    def _run(label: str, extra: Mapping[str, Any]) -> NestedOOFResult:
        merged = dict(kwargs)
        merged.update(extra)
        factory = make_trainer_factory(model_kind, **merged)
        return evaluate_nested_groupkfold_oof(
            df,
            trainer_factory=factory,
            n_splits=n_splits,
            targets=cols,
        )

    if include_full:
        full = _run("full", {})
        runs["full"] = full.to_report_dict(include_predictions=False)

    for name in names:
        spec = group_drop_spec(name)
        result = _run(f"drop_{name}", spec)
        runs[f"drop_{name}"] = result.to_report_dict(include_predictions=False)

    deltas: dict[str, Any] = {}
    if "full" in runs:
        full_mae = runs["full"].get("avg_mae_mm")
        for key, payload in runs.items():
            if key == "full":
                continue
            deltas[key] = {
                "avg_mae_mm_delta_vs_full": (
                    None
                    if full_mae is None or payload.get("avg_mae_mm") is None
                    else float(payload["avg_mae_mm"]) - float(full_mae)
                ),
                "allow_production_drop": False,
            }

    return {
        "protocol": "nested_groupkfold_ablation",
        "model_kind": model_kind,
        "n_splits": n_splits,
        "runs": runs,
        "delta_vs_full": deltas,
        "allow_production_drop": False,
        "note": PRODUCTION_DROP_FORBIDDEN_NOTE,
        "groups_tested": names,
        "na_pop_shift_expected_count": 6,
        "na_sup_affine_expected_count": 42,
    }
