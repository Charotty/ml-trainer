"""Optional linear correction of an anatomy feature onto the Excel scale.

``manual ≈ slope * auto + intercept``. The extractor applies
``config/anatomy_calibration.json`` when that file exists. The validation
script only proposes coefficients; it does not overwrite the config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CALIBRATION_PATH = REPO_ROOT / "config" / "anatomy_calibration.json"


def load_calibration(path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    path = Path(path) if path is not None else DEFAULT_CALIBRATION_PATH
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    cleaned: Dict[str, Dict[str, float]] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        try:
            cleaned[str(key)] = {
                "slope": float(value["slope"]),
                "intercept": float(value["intercept"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return cleaned


def apply_feature_calibration(
    features: Dict[str, object],
    calibration: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, object]:
    calib = calibration if calibration is not None else load_calibration()
    if not calib:
        return features
    out = dict(features)
    for key, coeff in calib.items():
        value = out.get(key)
        if value is None:
            continue
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not np.isfinite(number):
            continue
        slope = float(coeff["slope"])
        intercept = float(coeff["intercept"])
        out[key] = slope * number + intercept
        out[f"{key}_calibrated"] = True
    return out


def suggest_calibration(
    manual: np.ndarray,
    auto: np.ndarray,
    *,
    min_pairs: int = 8,
) -> Optional[Dict[str, float]]:
    """Fit manual ~ slope * auto + intercept. None when the cloud is too small."""
    x = np.asarray(auto, dtype=float)
    y = np.asarray(manual, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < min_pairs:
        return None
    x = x[ok]
    y = y[ok]
    if float(np.std(x)) < 1e-6:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept)}
