"""Population trends from na_spine (supine) and na_boku (lateral) CT extracts.

Clinical Vybor xlsx rows are the ONLY paired displacement labels for ensemble
training. na_spine / na_boku never contribute per-patient lateral coordinates
or proxy delta targets — they supply cohort-level value trends:

  * population shift priors  — median(lateral) − median(supine) per kidney axis
  * supine z-scores          — how a clinical row compares to na_spine cohort

KiTS19 is optional and not used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.phase1_schema import BASE_FEATURES, normalize_dataframe

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPINE_PATH = REPO_ROOT / "data" / "harmonized" / "na_spine_full_aligned.csv"
DEFAULT_BOKU_PATH = REPO_ROOT / "data" / "harmonized" / "na_boku_full_aligned.csv"
FALLBACK_SPINE_PATH = REPO_ROOT / "data" / "na_spine_full.csv"
FALLBACK_BOKU_PATH = REPO_ROOT / "data" / "na_boku_full.bak.csv"

TREND_COLUMNS = [
    c
    for c in BASE_FEATURES
    if c.startswith(("kidney_", "body_", "spine_", "body_com"))
]

POP_SHIFT_AXES = ("x", "y", "z")
POP_SHIFT_SIDES = ("left", "right")


def _resolve_path(primary: Path, fallback: Path) -> Path | None:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return None


def _robust_stats(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"median": np.nan, "mad": np.nan, "q10": np.nan, "q90": np.nan, "n": 0}
    med = float(s.median())
    mad = float((s - med).abs().median())
    if mad < 1e-9:
        mad = float(s.std()) if float(s.std()) > 1e-9 else 1.0
    return {
        "median": med,
        "mad": mad,
        "q10": float(s.quantile(0.10)),
        "q90": float(s.quantile(0.90)),
        "n": int(len(s)),
    }


@dataclass
class NaTrendStore:
    """Cohort statistics fitted on na_spine + na_boku (no Vybor labels)."""

    supine_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    lateral_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    population_shift: dict[str, float] = field(default_factory=dict)
    spine_rows: int = 0
    boku_rows: int = 0
    spine_path: str = ""
    boku_path: str = ""

    @classmethod
    def fit(
        cls,
        *,
        spine_path: Path | str | None = None,
        boku_path: Path | str | None = None,
    ) -> NaTrendStore:
        sp_p = _resolve_path(
            Path(spine_path) if spine_path else DEFAULT_SPINE_PATH,
            FALLBACK_SPINE_PATH,
        )
        bk_p = _resolve_path(
            Path(boku_path) if boku_path else DEFAULT_BOKU_PATH,
            FALLBACK_BOKU_PATH,
        )
        store = cls(
            spine_path=str(sp_p) if sp_p else "",
            boku_path=str(bk_p) if bk_p else "",
        )
        spine_df = normalize_dataframe(pd.read_csv(sp_p)) if sp_p else pd.DataFrame()
        boku_df = normalize_dataframe(pd.read_csv(bk_p)) if bk_p else pd.DataFrame()
        store.spine_rows = len(spine_df)
        store.boku_rows = len(boku_df)

        for col in TREND_COLUMNS:
            if col in spine_df.columns:
                store.supine_stats[col] = _robust_stats(spine_df[col])
            if col in boku_df.columns:
                store.lateral_stats[col] = _robust_stats(boku_df[col])

        for side in POP_SHIFT_SIDES:
            for axis in POP_SHIFT_AXES:
                col = f"kidney_{side}_center_{axis}_rel"
                sup = store.supine_stats.get(col, {}).get("median", np.nan)
                lat = store.lateral_stats.get(col, {}).get("median", np.nan)
                key = f"na_pop_shift_{side}_{axis}"
                store.population_shift[key] = (
                    float(lat - sup) if np.isfinite(sup) and np.isfinite(lat) else np.nan
                )

        return store

    def trend_feature_names(self) -> list[str]:
        names = list(self.population_shift.keys())
        for col in TREND_COLUMNS:
            if col in self.supine_stats:
                names.append(f"na_sup_z_{col}")
                names.append(f"na_sup_pct_{col}")
        return sorted(names)

    def attach(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cohort trend features; no per-patient lateral join."""
        out = df.copy()
        for key, val in self.population_shift.items():
            out[key] = val

        for col in TREND_COLUMNS:
            if col not in out.columns or col not in self.supine_stats:
                continue
            stats = self.supine_stats[col]
            med, mad = stats.get("median", np.nan), stats.get("mad", np.nan)
            q10, q90 = stats.get("q10", np.nan), stats.get("q90", np.nan)
            vals = pd.to_numeric(out[col], errors="coerce")
            if np.isfinite(med) and np.isfinite(mad) and mad > 0:
                out[f"na_sup_z_{col}"] = (vals - med) / mad
            else:
                out[f"na_sup_z_{col}"] = np.nan
            if np.isfinite(q10) and np.isfinite(q90) and q90 > q10:
                out[f"na_sup_pct_{col}"] = (vals - q10) / (q90 - q10)
            else:
                out[f"na_sup_pct_{col}"] = np.nan
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "spine_rows": self.spine_rows,
            "boku_rows": self.boku_rows,
            "spine_path": self.spine_path,
            "boku_path": self.boku_path,
            "population_shift": self.population_shift,
            "n_supine_stats": len(self.supine_stats),
            "n_lateral_stats": len(self.lateral_stats),
            "trend_features": len(self.trend_feature_names()),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "supine_stats": self.supine_stats,
            "lateral_stats": self.lateral_stats,
            "population_shift": self.population_shift,
            "spine_rows": self.spine_rows,
            "boku_rows": self.boku_rows,
            "spine_path": self.spine_path,
            "boku_path": self.boku_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NaTrendStore:
        return cls(
            supine_stats=dict(payload.get("supine_stats") or {}),
            lateral_stats=dict(payload.get("lateral_stats") or {}),
            population_shift=dict(payload.get("population_shift") or {}),
            spine_rows=int(payload.get("spine_rows") or 0),
            boku_rows=int(payload.get("boku_rows") or 0),
            spine_path=str(payload.get("spine_path") or ""),
            boku_path=str(payload.get("boku_path") or ""),
        )


def attach_na_trend_features(
    df: pd.DataFrame,
    store: NaTrendStore | None,
) -> pd.DataFrame:
    if store is None:
        return df
    return store.attach(df)
