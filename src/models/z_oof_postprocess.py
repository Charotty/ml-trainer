"""Honest Z calibration applied to cached nested-CV OOF predictions.

The span/lordosis calibrator (``SideZCalibrator``) must be fitted on
out-of-fold raw predictions: fitting on in-sample predictions of a model that
memorizes the training rows (RF/GBT) leaves nothing to correct, which is why
the gated fit returned ``None`` for every production run.

``calibrate_oof_z`` runs a second GroupKFold over the OOF predictions: for each
fold the gated calibrator is fitted on the other folds (with its own inner
GroupKFold gate) and applied to the held-out fold, so the reported MAE never
sees the rows it is evaluated on.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.models.z_calibrator_oof import SideZCalibrator, fit_calibrator_oof_gated

Z_SIDES = ("left", "right")


def calibrate_oof_z(
    frame: pd.DataFrame,
    oof_pred: pd.DataFrame,
    groups: Sequence[str],
    *,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Return OOF predictions with Z columns replaced by nested-calibrated values."""
    frame = frame.reset_index(drop=True)
    out = oof_pred.reset_index(drop=True).copy()
    groups_arr = np.asarray(groups).astype(str)
    usage: dict[str, dict[str, int]] = {}
    for side in Z_SIDES:
        target = f"kidney_{side}_delta_z"
        if target not in out.columns or target not in frame.columns:
            continue
        raw = out[target].to_numpy(dtype=float)
        y = frame[target].to_numpy(dtype=float)
        mask = np.isfinite(raw) & np.isfinite(y)
        idx = np.where(mask)[0]
        cal = raw.copy()
        used = 0
        splits = min(n_splits, len(np.unique(groups_arr[idx])))
        if splits >= 2:
            for tr, va in GroupKFold(n_splits=splits).split(idx, groups=groups_arr[idx]):
                tr_i, va_i = idx[tr], idx[va]
                fitted = fit_calibrator_oof_gated(
                    SideZCalibrator(side=side),
                    frame.iloc[tr_i].reset_index(drop=True),
                    raw[tr_i],
                    y[tr_i],
                    groups_arr[tr_i],
                )
                if fitted is not None:
                    used += 1
                    cal[va_i] = fitted.transform(
                        frame.iloc[va_i].reset_index(drop=True), raw[va_i]
                    )
        usage[target] = {"folds_calibrated": used, "folds": int(splits)}
        out[target] = cal
    return out, usage


def fit_final_calibrators(
    frame: pd.DataFrame,
    oof_pred: pd.DataFrame,
    groups: Sequence[str],
) -> dict[str, SideZCalibrator | None]:
    """Gated calibrators for the final model, fitted on OOF raw predictions."""
    frame = frame.reset_index(drop=True)
    oof_pred = oof_pred.reset_index(drop=True)
    out: dict[str, SideZCalibrator | None] = {}
    for side in Z_SIDES:
        target = f"kidney_{side}_delta_z"
        if target not in oof_pred.columns:
            out[side] = None
            continue
        out[side] = fit_calibrator_oof_gated(
            SideZCalibrator(side=side),
            frame,
            oof_pred[target].to_numpy(dtype=float),
            frame[target].to_numpy(dtype=float),
            np.asarray(groups).astype(str),
        )
    return out
