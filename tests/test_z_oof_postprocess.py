import numpy as np
import pandas as pd

from src.models.nested_cv import aggregate_per_seed, per_seed_summary, summarize_oof_with_clinical
from src.models.z_oof_postprocess import calibrate_oof_z, fit_final_calibrators
from src.features.phase1_schema import TARGET_NAMES


def _frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    span = rng.uniform(40, 140, n)
    lord = rng.uniform(20, 60, n)
    zrel = rng.normal(0, 5, n)
    df = pd.DataFrame(
        {
            "full_name": [f"P{i}" for i in range(n)],
            "kidney_left_z_span_supine_mm": span,
            "kidney_right_z_span_supine_mm": span,
            "lumbar_lordosis_deg": lord,
            "kidney_left_center_z_rel": zrel,
            "kidney_right_center_z_rel": zrel,
        }
    )
    anchor = span * 0.15 - lord * 0.08 - zrel * 0.35
    for t in TARGET_NAMES:
        df[t] = rng.normal(0, 3, n)
    df["kidney_left_delta_z"] = anchor + rng.normal(0, 1, n)
    df["kidney_right_delta_z"] = anchor + rng.normal(0, 1, n)
    return df


def test_calibrate_oof_z_is_nested_and_shapes_match():
    df = _frame()
    oof = pd.DataFrame({t: np.zeros(len(df)) for t in TARGET_NAMES})
    cal, usage = calibrate_oof_z(df, oof, df["full_name"].values)
    assert cal.shape == oof.shape
    assert set(usage) == {"kidney_left_delta_z", "kidney_right_delta_z"}
    # Non-Z columns untouched.
    assert np.allclose(cal["kidney_left_delta_x"], 0.0)
    m_raw = summarize_oof_with_clinical(df, oof, list(TARGET_NAMES))
    m_cal = summarize_oof_with_clinical(df, cal, list(TARGET_NAMES))
    assert m_cal["z_avg_mae_mm"] <= m_raw["z_avg_mae_mm"] + 1e-9
    agg = aggregate_per_seed([per_seed_summary(m_cal, 0)])
    assert np.isfinite(agg["avg_mae_mm"]["mean"])


def test_fit_final_calibrators_returns_sides():
    df = _frame()
    oof = pd.DataFrame({t: np.zeros(len(df)) for t in TARGET_NAMES})
    cals = fit_final_calibrators(df, oof, df["full_name"].values)
    assert set(cals) == {"left", "right"}
