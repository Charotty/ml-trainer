"""Fold-safe encoding of sex / body_type without a false linear order.

Unknown demographics stay missing (NaN). They are never replaced with 0 / 50 / 25
before the persisted imputer. Missingness can be exposed as explicit indicators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Canonical codes already used in Vybor / kits19 extracts. 0 is a real body_type
# (normosthenic), not "unknown". Unknown sex must NOT be coded as 0.
SEX_CODES: tuple[float, ...] = (1.0, 2.0)
BODY_TYPE_CODES: tuple[float, ...] = (0.0, 1.0, 2.0)
CATEGORICAL_COLS: tuple[str, ...] = ("sex", "body_type")
CONTINUOUS_MISS_COLS: tuple[str, ...] = ("age", "bmi", "has_previous_surgery")
ALL_MISS_COLS: tuple[str, ...] = ("sex", "age", "bmi", "body_type", "has_previous_surgery")


def _is_na(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def dummy_column_name(col: str, code: float) -> str:
    if float(code).is_integer():
        return f"{col}_code_{int(code)}"
    return f"{col}_code_{code}"


def missing_column_name(col: str) -> str:
    return f"{col}_missing"


@dataclass
class FoldCategoricalEncoder:
    """One-hot encode sex/body_type on the train fold; transform val/inference."""

    sex_codes: tuple[float, ...] = SEX_CODES
    body_type_codes: tuple[float, ...] = BODY_TYPE_CODES
    add_missing_indicators: bool = True
    drop_original: bool = True
    fitted_: bool = False
    dummy_names: list[str] = field(default_factory=list)
    missing_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sex_codes": list(self.sex_codes),
            "body_type_codes": list(self.body_type_codes),
            "add_missing_indicators": self.add_missing_indicators,
            "drop_original": self.drop_original,
            "fitted_": self.fitted_,
            "dummy_names": list(self.dummy_names),
            "missing_names": list(self.missing_names),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> FoldCategoricalEncoder:
        if not payload:
            return cls()
        enc = cls(
            sex_codes=tuple(float(x) for x in payload.get("sex_codes") or SEX_CODES),
            body_type_codes=tuple(
                float(x) for x in payload.get("body_type_codes") or BODY_TYPE_CODES
            ),
            add_missing_indicators=bool(payload.get("add_missing_indicators", True)),
            drop_original=bool(payload.get("drop_original", True)),
        )
        enc.fitted_ = bool(payload.get("fitted_", False))
        enc.dummy_names = [str(n) for n in payload.get("dummy_names") or []]
        enc.missing_names = [str(n) for n in payload.get("missing_names") or []]
        return enc

    def _codes_for(self, col: str) -> tuple[float, ...]:
        if col == "sex":
            return self.sex_codes
        if col == "body_type":
            return self.body_type_codes
        return ()

    def fit(self, df: pd.DataFrame) -> FoldCategoricalEncoder:
        dummy_names: list[str] = []
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            for code in self._codes_for(col):
                dummy_names.append(dummy_column_name(col, code))
        self.dummy_names = dummy_names
        self.missing_names = (
            [missing_column_name(c) for c in ALL_MISS_COLS if c in df.columns]
            if self.add_missing_indicators
            else []
        )
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("FoldCategoricalEncoder.transform called before fit")
        out = df.copy()
        for col in CATEGORICAL_COLS:
            codes = self._codes_for(col)
            if col not in out.columns:
                for code in codes:
                    name = dummy_column_name(col, code)
                    if name in self.dummy_names:
                        out[name] = 0.0
                continue
            numeric = pd.to_numeric(out[col], errors="coerce")
            for code in codes:
                name = dummy_column_name(col, code)
                if name in self.dummy_names:
                    out[name] = (numeric - code).abs() < 1e-9
                    out[name] = out[name].astype(float)
                    out.loc[numeric.isna(), name] = 0.0
            if self.drop_original:
                out = out.drop(columns=[col])
        if self.add_missing_indicators:
            for col in ALL_MISS_COLS:
                name = missing_column_name(col)
                if name not in self.missing_names:
                    continue
                if col in df.columns:
                    out[name] = pd.to_numeric(df[col], errors="coerce").isna().astype(float)
                else:
                    out[name] = 0.0
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def extra_feature_names(self) -> list[str]:
        return list(self.dummy_names) + list(self.missing_names)


def encode_sex_label(value: object) -> float:
    """Map a raw sex label to the canonical 1/2 code, else NaN (never 0)."""
    if _is_na(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"", "unknown", "nan", "none", "o", "other"}:
        return np.nan
    if text in {"м", "m", "male", "1", "1.0"}:
        return 1.0
    if text in {"ж", "f", "female", "2", "2.0"}:
        return 2.0
    try:
        number = float(text)
    except ValueError:
        return np.nan
    if abs(number - 1.0) < 1e-9:
        return 1.0
    if abs(number - 2.0) < 1e-9:
        return 2.0
    return np.nan


def parse_dicom_age(age: object) -> float:
    """Parse DICOM PatientAge (nY/nM/nW/nD). Missing → NaN, never 50."""
    import re

    if _is_na(age):
        return np.nan
    age_str = str(age).strip()
    if not age_str:
        return np.nan
    unit_match = re.search(r"(\d+(?:\.\d+)?)([YMWD])", age_str, flags=re.IGNORECASE)
    if unit_match:
        age_num = float(unit_match.group(1))
        unit = unit_match.group(2).upper()
        if unit == "Y":
            return age_num
        if unit == "M":
            return age_num / 12.0
        if unit == "W":
            return age_num / 52.0
        if unit == "D":
            return age_num / 365.0
        return age_num
    num_match = re.search(r"(\d+(?:\.\d+)?)", age_str)
    if not num_match:
        return np.nan
    return float(num_match.group(1))


def bmi_from_weight_height(weight_kg: object, height_m: object) -> float:
    """BMI from DICOM weight/size. Missing → NaN, never 25."""
    try:
        w = float(weight_kg) if not _is_na(weight_kg) else float("nan")
        h = float(height_m) if not _is_na(height_m) else float("nan")
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(w) or not np.isfinite(h) or h <= 0:
        return np.nan
    return w / (h * h)
