"""Regression tests for PredictorBundle na_trends restore and RF fallback policy."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "scripts" / "validation"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(VALIDATION) not in sys.path:
    sys.path.insert(0, str(VALIDATION))

from common import (  # noqa: E402
    DEFAULT_MODEL_PATH_STR,
    LEGACY_MODEL_NAME,
    build_or_load_predictor,
    load_model_bundle,
    warn_if_legacy_model,
)
from src.features.na_trend_features import (  # noqa: E402
    NaTrendStore,
    attach_na_trend_features,
)
from src.features.phase1_schema import BASE_FEATURES, TARGET_NAMES  # noqa: E402


def _minimal_store() -> NaTrendStore:
    return NaTrendStore(
        supine_stats={
            "kidney_left_center_x_rel": {
                "median": 10.0,
                "mad": 2.0,
                "q10": 5.0,
                "q90": 15.0,
                "n": 3,
            }
        },
        lateral_stats={
            "kidney_left_center_x_rel": {
                "median": 14.0,
                "mad": 2.0,
                "q10": 8.0,
                "q90": 20.0,
                "n": 3,
            }
        },
        population_shift={"na_pop_shift_left_x": 4.0},
        spine_rows=3,
        boku_rows=3,
        include_kits=False,
    )


def _toy_payload(store: NaTrendStore) -> dict:
    feature_names = list(BASE_FEATURES) + store.trend_feature_names()
    return {
        "models": {t: object() for t in TARGET_NAMES},
        "scaler": StandardScaler(),
        "imputer": None,
        "feature_names": feature_names,
        "target_names": list(TARGET_NAMES),
        "enrichment_mode": "na_trends",
        "na_trend_store": store.to_dict(),
        "left_z_calibrator": {"kind": "identity"},
        "right_z_calibrator": {"kind": "identity"},
        "z_head": "ensemble",
        "z_driver_names": None,
        "training_meta": {
            "clinical_only": True,
            "calibrators": "oof_gated_supine_only",
        },
    }


def _clinical_row() -> pd.DataFrame:
    row = {c: float(i + 1) for i, c in enumerate(BASE_FEATURES)}
    row.update({t: 1.0 for t in TARGET_NAMES})
    return pd.DataFrame([row])


def test_load_model_bundle_restores_enrichment_and_store(tmp_path: Path):
    store = _minimal_store()
    path = tmp_path / "honest_bundle.pkl"
    joblib.dump(_toy_payload(store), path)

    bundle = load_model_bundle(path)
    assert bundle.enrichment_mode == "na_trends"
    assert bundle.na_trend_store is not None
    assert bundle.na_trend_store["population_shift"]["na_pop_shift_left_x"] == 4.0

    restored = NaTrendStore.from_dict(bundle.na_trend_store)
    clinical = _clinical_row()
    expected = attach_na_trend_features(clinical, store)
    got = attach_na_trend_features(clinical, restored)
    assert "na_pop_shift_left_x" in got.columns
    assert "na_sup_z_kidney_left_center_x_rel" in got.columns
    pd.testing.assert_frame_equal(got, expected)


def test_honest_style_payload_required_keys_roundtrip(tmp_path: Path):
    """clinical_honest joblib shape: enrichment, trends, imputer/scaler/models, calibrators."""
    store = _minimal_store()
    payload = _toy_payload(store)
    required = {
        "enrichment_mode",
        "na_trend_store",
        "imputer",
        "scaler",
        "models",
        "feature_names",
        "target_names",
        "left_z_calibrator",
        "right_z_calibrator",
        "z_head",
    }
    assert required.issubset(payload.keys())
    assert payload["enrichment_mode"] == "na_trends"
    assert isinstance(payload["na_trend_store"], dict)
    assert set(payload["models"].keys()) == set(TARGET_NAMES)

    path = tmp_path / "clinical_honest_style.pkl"
    joblib.dump(payload, path)
    bundle = load_model_bundle(path)

    assert bundle.enrichment_mode == "na_trends"
    assert bundle.na_trend_store is not None
    assert bundle.scaler is not None
    assert bundle.imputer is None
    assert set(bundle.models.keys()) == set(TARGET_NAMES)
    assert bundle.left_z_calibrator == {"kind": "identity"}
    assert bundle.right_z_calibrator == {"kind": "identity"}
    assert bundle.z_head == "ensemble"


def test_build_or_load_predictor_restores_na_trends(tmp_path: Path):
    store = _minimal_store()
    path = tmp_path / "honest_bundle.pkl"
    joblib.dump(_toy_payload(store), path)

    df = _clinical_row()
    bundle, _, _ = build_or_load_predictor(
        df, path, test_size=0.5, seed=0, holdout_eval=True
    )
    assert bundle.mode == "pretrained_adaptive_ensemble"
    assert bundle.enrichment_mode == "na_trends"
    assert bundle.na_trend_store["spine_rows"] == 3


def test_build_or_load_predictor_raises_on_corrupt_pkl(tmp_path: Path):
    bad = tmp_path / "corrupt.pkl"
    bad.write_bytes(b"not-a-valid-joblib-payload")
    df = _clinical_row()
    with pytest.raises(Exception):
        build_or_load_predictor(df, bad, test_size=0.5, seed=0, holdout_eval=True)


def test_build_or_load_predictor_raises_on_missing_path(tmp_path: Path):
    missing = tmp_path / "does_not_exist.pkl"
    df = _clinical_row()
    with pytest.raises(FileNotFoundError):
        build_or_load_predictor(df, missing, test_size=0.5, seed=0, holdout_eval=True)


def test_build_or_load_predictor_rf_only_when_model_path_none():
    rng = np.random.default_rng(0)
    n = 20
    data = {c: rng.normal(size=n) for c in BASE_FEATURES}
    data.update({t: rng.normal(size=n) for t in TARGET_NAMES})
    df = pd.DataFrame(data)

    bundle, train_df, eval_df = build_or_load_predictor(
        df, None, test_size=0.3, seed=0
    )
    assert bundle.mode == "fallback_random_forest"
    assert len(train_df) + len(eval_df) == n
    assert set(bundle.target_names) == set(TARGET_NAMES)


def test_warn_if_legacy_model():
    with pytest.warns(UserWarning, match="legacy"):
        warn_if_legacy_model(Path("models") / LEGACY_MODEL_NAME)


def test_default_model_path_is_clinical_honest():
    assert DEFAULT_MODEL_PATH_STR.endswith("adaptive_ensemble_clinical_honest.pkl")
    assert LEGACY_MODEL_NAME == "adaptive_ensemble.pkl"


def test_smoke_check_missing_model_fails(tmp_path: Path, monkeypatch):
    from smoke_check import main, parse_args

    missing_model = tmp_path / "missing_model.pkl"
    # Provide an existing dataset stub so only the model check fails.
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("a\n1\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke_check.py", "--dataset", str(dataset), "--model", str(missing_model)],
    )
    args = parse_args()
    assert args.model == str(missing_model)
    code = main()
    assert code == 2
