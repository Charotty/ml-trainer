#!/usr/bin/env python3
"""Evaluate a trained model on Variant A holdout patients (CT + xlsx)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.holdout import (  # noqa: E402
    DEFAULT_HOLDOUT_PATH,
    load_holdout_config,
    mae_holdout_patients,
    normalize_surname_key,
    plausibility_holdout_patients,
)
from src.data.xlsx_displacement_parser import (  # noqa: E402
    DEFAULT_XLSX_PATH,
    build_vybor_from_xlsx,
)
from src.features.phase1_schema import TARGET_NAMES, normalize_dataframe  # noqa: E402
from src.models.runtime import load_model_bundle, predict_from_bundle  # noqa: E402

PLAUSIBLE_ABS_MAX_MM = 80.0


def _find_case_base_features(case_id: str) -> dict[str, Any] | None:
    case_dir = ROOT / "data" / "cases" / case_id
    candidates = [
        case_dir / "artifacts" / "base_features.json",
        case_dir / "base_features.json",
    ]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    # Nested artifacts folders
    if case_dir.is_dir():
        for path in case_dir.rglob("base_features.json"):
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _load_report_base_features(report_rel: str | None, case_id: str) -> dict[str, Any] | None:
    paths = []
    if report_rel:
        paths.append(ROOT / report_rel)
    paths.append(ROOT / "reports" / f"report_{case_id}.json")
    for path in paths:
        if not path.is_file():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        base = doc.get("base_features")
        if isinstance(base, dict) and base:
            return base
    return None


def _xlsx_holdout_frame(xlsx: Path) -> pd.DataFrame:
    """Build canonical rows for holdout patients only (include_holdout)."""
    full = build_vybor_from_xlsx(xlsx, boku_path=None, exclude_surnames=[])
    # Filter to MAE-scored surnames present in the sheet.
    cfg = load_holdout_config()
    keys = {
        normalize_surname_key(p["surname"])
        for p in mae_holdout_patients(cfg)
    }
    for p in mae_holdout_patients(cfg):
        for alias in p.get("aliases") or []:
            keys.add(normalize_surname_key(alias))
    name_col = "full_name" if "full_name" in full.columns else "fio"
    mask = full[name_col].map(normalize_surname_key).isin(keys)
    return full.loc[mask].reset_index(drop=True)


def _predict_row(bundle, row: dict[str, Any] | pd.Series) -> dict[str, float | None]:
    df = normalize_dataframe(pd.DataFrame([dict(row)]))
    pred = predict_from_bundle(bundle, df)
    out: dict[str, float | None] = {}
    for t in TARGET_NAMES:
        if t not in pred.columns:
            out[t] = None
            continue
        val = float(pred[t].iloc[0])
        out[t] = None if not np.isfinite(val) else val
    return out


def _mae_against_truth(
    pred: dict[str, float | None],
    truth: MappingLike,
    *,
    skip_left: bool = False,
) -> dict[str, Any]:
    errs: list[float] = []
    per_target: dict[str, float | None] = {}
    for t in TARGET_NAMES:
        if skip_left and t.startswith("kidney_left_"):
            per_target[t] = None
            continue
        yt = truth.get(t) if hasattr(truth, "get") else truth[t]
        yp = pred.get(t)
        try:
            yt_f = float(yt) if yt is not None and str(yt) != "nan" else float("nan")
        except (TypeError, ValueError):
            yt_f = float("nan")
        if yp is None or not np.isfinite(yp) or not np.isfinite(yt_f):
            per_target[t] = None
            continue
        err = abs(float(yp) - yt_f)
        per_target[t] = err
        errs.append(err)
    return {
        "per_target_abs_err_mm": per_target,
        "mean_abs_err_mm": float(np.mean(errs)) if errs else float("nan"),
        "n_scored": len(errs),
    }


# typing helper alias
MappingLike = Any


def _plausibility(pred: dict[str, float | None]) -> dict[str, Any]:
    finite = {k: v for k, v in pred.items() if v is not None and np.isfinite(v)}
    issues: list[str] = []
    for k, v in finite.items():
        if abs(v) > PLAUSIBLE_ABS_MAX_MM:
            issues.append(f"{k}={v:.1f} outside ±{PLAUSIBLE_ABS_MAX_MM} mm")
    signs = {k: (1 if v > 0 else -1 if v < 0 else 0) for k, v in finite.items()}
    return {
        "n_finite": len(finite),
        "predictions": finite,
        "signs": signs,
        "issues": issues,
        "ok": len(issues) == 0 and len(finite) > 0,
    }


def evaluate_holdout(
    model_path: Path,
    *,
    xlsx: Path = DEFAULT_XLSX_PATH,
    holdout_path: Path = DEFAULT_HOLDOUT_PATH,
) -> dict[str, Any]:
    cfg = load_holdout_config(holdout_path)
    bundle = load_model_bundle(model_path)
    xlsx_frame = _xlsx_holdout_frame(xlsx) if xlsx.exists() else pd.DataFrame()
    name_col = "full_name" if "full_name" in xlsx_frame.columns else "fio"

    patients_out: list[dict[str, Any]] = []
    trio_errs: list[float] = []

    for patient in cfg.get("patients") or []:
        surname = patient.get("surname")
        case_id = str(patient.get("case_id") or "")
        score_mode = patient.get("score_mode")
        skip_left = "крема" in normalize_surname_key(surname) or "krem" in normalize_surname_key(
            surname
        )
        entry: dict[str, Any] = {
            "surname": surname,
            "case_id": case_id,
            "score_mode": score_mode,
            "ct": None,
            "xlsx": None,
        }

        base = _find_case_base_features(case_id)
        base_source = "data/cases"
        if base is None:
            base = _load_report_base_features(patient.get("report"), case_id)
            base_source = "reports"
        if base is not None:
            try:
                pred_ct = _predict_row(bundle, base)
                entry["ct"] = {
                    "features_source": base_source,
                    "predictions": pred_ct,
                    "plausibility": _plausibility(pred_ct),
                }
            except Exception as exc:  # noqa: BLE001 — report per-patient failure
                entry["ct"] = {"error": str(exc), "features_source": base_source}
        else:
            entry["ct"] = {"error": "base_features not found on disk or in reports"}

        if score_mode == "mae" and not xlsx_frame.empty:
            keys = {normalize_surname_key(surname)}
            for alias in patient.get("aliases") or []:
                keys.add(normalize_surname_key(alias))
            mask = xlsx_frame[name_col].map(normalize_surname_key).isin(keys)
            rows = xlsx_frame.loc[mask]
            if len(rows) == 0:
                entry["xlsx"] = {"error": "surname not found in xlsx after include-holdout parse"}
            else:
                row = rows.iloc[0]
                pred_x = _predict_row(bundle, row)
                truth = {t: row[t] for t in TARGET_NAMES if t in rows.columns}
                mae = _mae_against_truth(pred_x, truth, skip_left=skip_left)
                entry["xlsx"] = {
                    "predictions": pred_x,
                    "truth": {k: (None if not np.isfinite(float(v)) else float(v))
                              for k, v in truth.items()
                              if v is not None and str(v) != "nan"},
                    "error": mae,
                }
                if np.isfinite(mae["mean_abs_err_mm"]):
                    trio_errs.append(float(mae["mean_abs_err_mm"]))
        elif score_mode == "plausibility":
            if entry.get("ct") and entry["ct"].get("predictions"):
                entry["plausibility"] = entry["ct"]["plausibility"]

        patients_out.append(entry)

    report = {
        "model_path": str(model_path),
        "holdout_config": str(holdout_path),
        "xlsx": str(xlsx),
        "patients": patients_out,
        "holdout_trio_mean_mae_mm": float(np.mean(trio_errs)) if trio_errs else float("nan"),
        "holdout_trio_n": len(trio_errs),
        "delta_definition": "lateral middle minus supine middle",
    }
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Holdout evaluation for Variant A")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX_PATH)
    p.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT_PATH)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_holdout(args.model, xlsx=args.xlsx, holdout_path=args.holdout)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
