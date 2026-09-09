"""Block 2: nested GroupKFold, fit_final (no dummy val), honest proxy protocol."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ensemble import AdaptiveEnsembleTrainer  # noqa: E402
from src.features.phase1_schema import BASE_FEATURES, TARGET_NAMES  # noqa: E402
from src.models.artifact_manifest import apply_oof_to_manifest, oof_placeholder  # noqa: E402
from src.models.nested_cv import (  # noqa: E402
    evaluate_nested_groupkfold_oof,
    evaluate_nested_proxy_vs_honest,
)

SPIKE = 50_000.0
SPIKE_COL = "body_width_mm"


def _synthetic_clinical(
    n_patients: int = 6,
    *,
    seed: int = 0,
    spike_patient: str | None = "P05",
    targets: list[str] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    use_targets = list(targets or TARGET_NAMES[:2])
    rows = []
    for i in range(n_patients):
        pid = f"P{i:02d}"
        row = {col: float(100.0 + rng.normal(0, 2)) for col in BASE_FEATURES}
        row["full_name"] = pid
        row[SPIKE_COL] = 280.0 + i
        if spike_patient and pid == spike_patient:
            row[SPIKE_COL] = SPIKE
        for t in use_targets:
            row[t] = float(rng.normal(0.0, 2.0))
        rows.append(row)
    return pd.DataFrame(rows)


def _tiny_factory(**kwargs):
    params = {
        "enrichment_mode": "none",
        "estimator_profile": "tiny",
        "inner_n_splits": 2,
    }
    params.update(kwargs)
    return AdaptiveEnsembleTrainer(**params)


def test_nested_cv_does_not_leak_val_into_imputer_scaler() -> None:
    df = _synthetic_clinical()
    spike_id = "P05"
    result = evaluate_nested_groupkfold_oof(
        df,
        trainer_factory=_tiny_factory,
        n_splits=2,
        weight_mode="inner_groupkfold",
        targets=[c for c in TARGET_NAMES[:2] if c in df.columns],
    )
    val_folds = [fold for fold in result.folds if spike_id in fold["val_groups"]]
    assert val_folds, "expected the spiked patient to appear in at least one outer val fold"
    for fold in val_folds:
        stats = fold["preprocessor"]
        mean = stats["scaler_mean"][SPIKE_COL]
        imputed = stats["imputer_statistics"][SPIKE_COL]
        assert mean is not None and imputed is not None
        assert mean < 1_000, f"scaler.mean_ leaked val spike: {mean}"
        assert imputed < 1_000, f"imputer.statistics_ leaked val spike: {imputed}"
        assert abs(mean - SPIKE) > 10_000
    assert result.protocol == "nested_groupkfold"
    assert result.weight_mode == "inner_groupkfold"
    assert "avg_mae_mm" in result.metrics


def test_proxy_eval_does_not_call_honest_only_oof_as_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _synthetic_clinical(n_patients=6, spike_patient=None)

    def _boom(*_args, **_kwargs):
        raise AssertionError("honest-only OOF must not be used as a proxy metric")

    monkeypatch.setattr(
        "src.models.nested_cv.evaluate_nested_groupkfold_oof",
        _boom,
    )

    seen_teacher_groups: list[set[str]] = []

    def proxy_train_builder(train_clinical: pd.DataFrame, teacher) -> pd.DataFrame:
        assert teacher.trained_models, "teacher must be fit on outer-train before proxy rows"
        seen_teacher_groups.append(set(train_clinical["full_name"].astype(str)))
        extra = train_clinical.iloc[:1].copy()
        extra["full_name"] = extra["full_name"].astype(str) + "_proxy"
        extra["case_id"] = extra["full_name"]
        extra["sample_weight"] = 0.08
        extra["source"] = "KiTS19"
        clinical = train_clinical.copy()
        clinical["sample_weight"] = 1.0
        clinical["source"] = "Vybor"
        return pd.concat([clinical, extra], ignore_index=True)

    out = evaluate_nested_proxy_vs_honest(
        df,
        trainer_factory=_tiny_factory,
        proxy_train_builder=proxy_train_builder,
        n_splits=2,
        targets=[c for c in TARGET_NAMES[:2] if c in df.columns],
    )
    assert out["protocol"] == "nested_proxy_vs_honest"
    assert "clinical_proxy" in out and "clinical_honest" in out
    assert "per_target_mae_mm" in out["clinical_proxy"]
    all_ids = set(df["full_name"].astype(str))
    for fold, trained in zip(out["folds"], seen_teacher_groups):
        val_ids = set(fold["val_groups"])
        assert trained.isdisjoint(val_ids)
        assert trained <= (all_ids - val_ids)

    src = (ROOT / "scripts" / "validation" / "compare_proxy_vs_honest.py").read_text(
        encoding="utf-8"
    )
    assert "evaluate_groupkfold_oof" not in src
    proxy_src = (ROOT / "scripts" / "data" / "train_clinical_proxy.py").read_text(encoding="utf-8")
    assert "evaluate_groupkfold_oof" not in proxy_src


def test_proxy_builder_cannot_include_outer_val_patients() -> None:
    df = _synthetic_clinical(n_patients=6, spike_patient=None)

    def leaky_builder(train_clinical: pd.DataFrame, teacher) -> pd.DataFrame:
        del teacher
        return df.copy()

    with pytest.raises(ValueError, match="outer-validation patients"):
        evaluate_nested_proxy_vs_honest(
            df,
            trainer_factory=_tiny_factory,
            proxy_train_builder=leaky_builder,
            n_splits=2,
            targets=[c for c in TARGET_NAMES[:2] if c in df.columns],
        )


def test_fit_final_api_rejects_dummy_one_row_val_selection() -> None:
    sig = inspect.signature(AdaptiveEnsembleTrainer.fit_final)
    assert "X_test" not in sig.parameters
    assert "y_test" not in sig.parameters
    assert "X_val" not in sig.parameters
    assert "y_val" not in sig.parameters

    df = _synthetic_clinical(n_patients=6, spike_patient=None)
    trainer = _tiny_factory()
    X, y = trainer.prepare_training_data_fit(df)
    assert X is not None and y is not None
    groups = df["full_name"].astype(str).values
    trainer.fit_final(X, y, groups=groups)
    assert trainer.trained_models, "fit_final must produce models without a dummy val row"

    other = _tiny_factory()
    dummy = df.iloc[:1]
    X_tr, X_va, y_tr, y_va = other.prepare_training_data_split(df, dummy)
    with pytest.raises(ValueError, match="at least 2 validation rows"):
        other.train_and_evaluate_adaptive_ensembles(
            X_tr, X_va, y_tr, y_va, groups=groups, select_on_test=True
        )


def test_oof_placeholder_and_apply_to_manifest() -> None:
    empty = oof_placeholder(protocol="nested_groupkfold")
    assert empty["oof_status"] == "not_computed"
    assert empty["folds"] is None
    card = apply_oof_to_manifest(
        {"candidate_id": "x"},
        {
            "folds": [{"fold": 0, "n_train": 4, "n_val": 2}],
            "oof_predictions": {"group_ids": ["P00"], "targets": {}},
            "aggregated_metrics": {"avg_mae_mm": 1.0},
            "oof_protocol": "nested_groupkfold",
            "oof_status": "computed",
            "oof_weight_mode": "inner_groupkfold",
        },
    )
    assert card["oof_status"] == "computed"
    assert card["folds"][0]["n_val"] == 2
    assert card["aggregated_metrics"]["avg_mae_mm"] == 1.0
