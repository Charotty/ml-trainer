"""Align DICOM/LPS extract features to Vybor clinical-local coordinate frame.

Outputs are written under ``data/harmonized/`` — originals are never modified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.features.phase1_schema import BASE_FEATURES, TARGET_NAMES, normalize_dataframe
from src.features.ct_geometry import X_DISTRIBUTION_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARMONIZED_DIR = REPO_ROOT / "data" / "harmonized"
DEFAULT_REFERENCE_CSV = REPO_ROOT / "data" / "vybor_from_xlsx.csv"

# DICOM LPS relative Y is flipped vs Vybor table coordinates (posterior/anterior sign).
Y_SIGN_FLIP_SOURCES = frozenset({"dicom_lps", "na_spine", "na_boku", "dicom", "kits19"})

REL_COLUMNS = [
    f"kidney_{side}_center_{axis}_rel"
    for side in ("left", "right")
    for axis in ("x", "y", "z")
]

ABS_KIDNEY_COLUMNS = [
    f"kidney_{side}_center_{axis}"
    for side in ("left", "right")
    for axis in ("x", "y", "z")
]

SPINE_COLUMNS = ["spine_center_x", "spine_center_y", "spine_center_z"]
COM_COLUMNS = ["body_com_x", "body_com_y", "body_com_z"]

DISTANCE_COLUMNS = [
    "kidney_left_to_spine_distance",
    "kidney_right_to_spine_distance",
    "kidney_left_to_body_center_distance",
    "kidney_right_to_body_center_distance",
]

SCALE_FEATURES = [
    c
    for c in BASE_FEATURES
    if c not in DISTANCE_COLUMNS and c not in SPINE_COLUMNS + COM_COLUMNS
]

# Signed laterality (clinical-frame X). Must not be IQR-rescaled when the
# source IQR collapsed to ~0 (abs-from-kidney-midpoint degeneration).
SIGNED_X_COLUMNS = list(X_DISTRIBUTION_FEATURES)


@dataclass
class ReferenceStats:
    """Robust per-feature reference (Vybor clinical-local)."""

    source: str
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    spine_anchor: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "feature_stats": self.feature_stats,
            "spine_anchor": self.spine_anchor,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ReferenceStats":
        return cls(
            source=str(data.get("source", "vybor")),
            feature_stats=dict(data.get("feature_stats", {})),  # type: ignore[arg-type]
            spine_anchor=dict(data.get("spine_anchor", {})),  # type: ignore[arg-type]
        )


def _robust_stats(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0, "iqr": 1.0, "n_valid": 0.0}
    p25, p50, p75 = np.percentile(s, [25, 50, 75])
    iqr = float(p75 - p25)
    if iqr < 1e-6:
        iqr = float(s.std()) if float(s.std()) > 1e-6 else 1.0
    return {
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "iqr": float(iqr),
        "n_valid": float(len(s)),
    }


def _stats_usable_for_rescale(stats: Mapping[str, float], *, min_valid: int = 5) -> bool:
    return int(stats.get("n_valid", 0)) >= min_valid


def _signed_x_source_degenerate(stats: Mapping[str, float], *, eps: float = 1e-3) -> bool:
    """True when a signed X column has collapsed to ~zero (old abs-midpoint bug).

    Uses p25/p50/p75 rather than IQR: ``_robust_stats`` floors a zero IQR to 1.0.
    """
    p25 = float(stats.get("p25", 0.0) or 0.0)
    p50 = float(stats.get("p50", 0.0) or 0.0)
    p75 = float(stats.get("p75", 0.0) or 0.0)
    return abs(p25) < eps and abs(p50) < eps and abs(p75) < eps


def build_reference_stats(reference_df: pd.DataFrame, *, source: str = "vybor") -> ReferenceStats:
    ref = normalize_dataframe(reference_df)
    stats: Dict[str, Dict[str, float]] = {}
    for col in SCALE_FEATURES:
        if col in ref.columns:
            stats[col] = _robust_stats(ref[col])

    anchor = {}
    for col in SPINE_COLUMNS:
        if col in ref.columns:
            anchor[col] = float(ref[col].median(skipna=True))
    for col in COM_COLUMNS:
        if col in ref.columns:
            anchor[col] = float(ref[col].median(skipna=True))

    return ReferenceStats(source=source, feature_stats=stats, spine_anchor=anchor)


def save_reference_stats(stats: ReferenceStats, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")


def load_reference_stats(path: Path) -> ReferenceStats:
    return ReferenceStats.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _infer_source_kind(df: pd.DataFrame, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if "dicom_cohort" in df.columns:
        cohort = df["dicom_cohort"].dropna().astype(str).str.lower()
        if cohort.str.contains("boku").any():
            return "na_boku"
        if cohort.str.contains("spine").any():
            return "na_spine"
        return "dicom_lps"
    if "source" in df.columns and df["source"].astype(str).str.contains("Vybor", case=False).any():
        return "vybor"
    return "dicom_lps"


def _apply_y_axis_flip(df: pd.DataFrame) -> pd.DataFrame:
    """Flip LPS Y sign on kidney-relative and body offsets to match Vybor."""
    out = df.copy()
    for col in REL_COLUMNS:
        if col in out.columns and col.endswith("_y_rel"):
            out[col] = -pd.to_numeric(out[col], errors="coerce")

    if all(c in out.columns for c in SPINE_COLUMNS + COM_COLUMNS):
        for s_col, c_col in zip(SPINE_COLUMNS, COM_COLUMNS):
            spine = pd.to_numeric(out[s_col], errors="coerce")
            com = pd.to_numeric(out[c_col], errors="coerce")
            # Preserve offset com - spine, flip Y offset only
            offset_y = com - spine
            out[c_col] = spine + offset_y * (-1.0)

    for col in ABS_KIDNEY_COLUMNS:
        if col in out.columns and col.endswith("_y"):
            if all(c in out.columns for c in SPINE_COLUMNS):
                spine_y = pd.to_numeric(out["spine_center_y"], errors="coerce")
                kidney_y = pd.to_numeric(out[col], errors="coerce")
                rel_y = kidney_y - spine_y
                out[col] = spine_y - rel_y
    return out


def _robust_rescale_column(
    values: pd.Series,
    src_stats: Dict[str, float],
    ref_stats: Dict[str, float],
) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    src_iqr = src_stats.get("iqr", 1.0) or 1.0
    ref_iqr = ref_stats.get("iqr", 1.0) or 1.0
    src_p50 = src_stats.get("p50", 0.0)
    ref_p50 = ref_stats.get("p50", 0.0)
    return (v - src_p50) / src_iqr * ref_iqr + ref_p50


def _recompute_distances(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DISTANCE_COLUMNS:
        if col in out.columns:
            out[col] = np.nan
    return normalize_dataframe(out, inplace=False)


def _apply_spine_anchor(df: pd.DataFrame, anchor: Mapping[str, float]) -> pd.DataFrame:
    """Shift spine/body_com into Vybor-like absolute coordinates; keep rel coords."""
    out = df.copy()
    for col in SPINE_COLUMNS:
        if col in anchor and col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            shift = float(anchor[col]) - float(values.median(skipna=True))
            out[col] = values + shift
    for col in COM_COLUMNS:
        spine_key = col.replace("body_com", "spine_center")
        if spine_key in out.columns and col in out.columns:
            # Vybor table: body_com tracks spine_center after harmonization.
            out[col] = pd.to_numeric(out[spine_key], errors="coerce")
    return out


def harmonize_dataframe(
    df: pd.DataFrame,
    reference: ReferenceStats,
    *,
    source_kind: Optional[str] = None,
    src_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Map extract rows into Vybor-compatible clinical-local feature space."""
    kind = _infer_source_kind(df, source_kind)
    out = normalize_dataframe(df.copy())

    if kind in ("vybor", "excel", "clinical"):
        out["harmonization_source"] = kind
        out["harmonization_applied"] = "identity"
        return out

    out["harmonization_source"] = kind
    out["harmonization_applied"] = "dicom_lps_to_vybor"

    out = _apply_y_axis_flip(out)

    if src_stats is None:
        src_stats = {}
        for col in SCALE_FEATURES:
            if col in out.columns:
                src_stats[col] = _robust_stats(out[col])

    for col in SCALE_FEATURES:
        if col not in out.columns:
            continue
        if col not in reference.feature_stats or col not in src_stats:
            continue
        if not _stats_usable_for_rescale(reference.feature_stats[col]):
            continue
        if not _stats_usable_for_rescale(src_stats[col]):
            continue
        if col in SIGNED_X_COLUMNS and _signed_x_source_degenerate(src_stats[col]):
            # Do not map a collapsed-to-zero X axis onto the train median.
            continue
        out[col] = _robust_rescale_column(
            out[col], src_stats[col], reference.feature_stats[col]
        )

    out = _apply_spine_anchor(out, reference.spine_anchor)
    out = _recompute_distances(out)
    return out


def harmonize_file(
    input_path: Path,
    output_path: Path,
    reference: ReferenceStats,
    *,
    source_kind: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    out = harmonize_dataframe(df, reference, source_kind=source_kind)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def alignment_report(
    reference_df: pd.DataFrame,
    aligned_df: pd.DataFrame,
    *,
    features: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compare medians: reference vs aligned (for QA CSV)."""
    feats = list(features or BASE_FEATURES)
    ref = normalize_dataframe(reference_df)
    rows = []
    for col in feats:
        if col not in ref.columns or col not in aligned_df.columns:
            continue
        r = pd.to_numeric(ref[col], errors="coerce").dropna()
        a = pd.to_numeric(aligned_df[col], errors="coerce").dropna()
        if len(r) == 0 or len(a) == 0:
            continue
        rows.append(
            {
                "feature": col,
                "ref_median": float(r.median()),
                "aligned_median": float(a.median()),
                "delta_median": float(a.median() - r.median()),
                "ref_iqr": float(np.percentile(r, 75) - np.percentile(r, 25)),
                "aligned_iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
            }
        )
    return pd.DataFrame(rows)


def compare_x_feature_distributions(
    reference_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    *,
    median_iqr_factor: float = 5.0,
    iqr_ratio_max: float = 8.0,
    features: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Gate train vs inference X geometry (median / IQR).

    Returns a dict with ``wild_divergence`` (numerical mismatch), ``ood_flagged``
    (inference rows already marked ``geometry_ood_x``), and ``compatible``
    which is True when there is no wild divergence *or* the mismatch is
    explicitly OOD-flagged. Tests should fail when ``compatible`` is False.
    """
    cols = list(features or X_DISTRIBUTION_FEATURES)
    ref = normalize_dataframe(reference_df)
    inf = infer_df.copy()
    diverged: List[str] = []
    per_feature: Dict[str, Dict[str, float]] = {}
    for col in cols:
        if col not in ref.columns or col not in inf.columns:
            continue
        r = pd.to_numeric(ref[col], errors="coerce").dropna()
        a = pd.to_numeric(inf[col], errors="coerce").dropna()
        if len(r) == 0 or len(a) == 0:
            continue
        ref_iqr = float(np.percentile(r, 75) - np.percentile(r, 25))
        if ref_iqr < 1e-6:
            ref_iqr = float(r.std()) if float(r.std()) > 1e-6 else 1.0
        inf_iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
        if inf_iqr < 1e-6:
            inf_iqr = 0.0
        delta_median = float(a.median() - r.median())
        iqr_ratio = (inf_iqr / ref_iqr) if ref_iqr > 0 else float("inf")
        per_feature[col] = {
            "ref_median": float(r.median()),
            "infer_median": float(a.median()),
            "delta_median": delta_median,
            "ref_iqr": ref_iqr,
            "infer_iqr": inf_iqr,
            "iqr_ratio": float(iqr_ratio),
        }
        if abs(delta_median) > median_iqr_factor * ref_iqr:
            diverged.append(f"{col}: |delta_median|={abs(delta_median):.3f} > {median_iqr_factor}×IQR")
        if iqr_ratio > iqr_ratio_max:
            diverged.append(f"{col}: IQR ratio {iqr_ratio:.2f} > {iqr_ratio_max}")

    ood_flagged = False
    if "geometry_ood_x" in inf.columns:
        flags = inf["geometry_ood_x"]
        if flags.dtype == bool or flags.dtype == object:
            ood_flagged = bool(pd.Series(flags).astype(bool).any())
        else:
            ood_flagged = bool(pd.to_numeric(flags, errors="coerce").fillna(0).astype(bool).any())

    wild = bool(diverged)
    return {
        "wild_divergence": wild,
        "ood_flagged": ood_flagged,
        "compatible": (not wild) or ood_flagged,
        "diverged": diverged,
        "features": per_feature,
    }


def default_harmonization_manifest(
    *,
    reference_path: Path,
    outputs: Dict[str, str],
    transforms: List[str],
) -> dict:
    return {
        "target_frame": "vybor_clinical_local",
        "reference_csv": str(reference_path),
        "transforms": transforms,
        "outputs": outputs,
        "notes": (
            "DICOM/LPS extracts: Y-axis sign flip + robust IQR rescale per BASE feature "
            "to Vybor; signed X rel/sep are not rescaled when their source IQR has "
            "collapsed to zero; spine/body_com anchored to Vybor medians; distances recomputed."
        ),
    }
