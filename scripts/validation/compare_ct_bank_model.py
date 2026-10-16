#!/usr/bin/env python3
"""Compare the CT-bank model with the current serving model.

OOF numbers come from the training reports. Holdout numbers are predictions
on the five saved cases, with CT-profile predictors attached the same way
training attaches them, then the same Z policy as the API.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.cases.predictor import finalize_case_predictions  # noqa: E402
from src.data.ct_profile_bank import attach_predictors, load_bank, load_profile  # noqa: E402
from src.data.holdout import normalize_surname_key  # noqa: E402
from src.data.xlsx_displacement_parser import parse_xlsx_raw_table  # noqa: E402
from src.models.runtime import RuntimePredictor  # noqa: E402

OLD_MODEL = ROOT / "models" / "adaptive_ensemble_clinical_honest.pkl"
NEW_MODEL = ROOT / "models" / "adaptive_ensemble_clinical_honest_ct_bank.pkl"
OLD_REPORT = ROOT / "results" / "validation_runs" / "clinical_honest_ensemble_20261015" / "metrics" / "clinical_honest_report.json"
NEW_REPORT = ROOT / "results" / "validation_runs" / "clinical_honest_ensemble_20261016" / "metrics" / "clinical_honest_report.json"
XLSX = ROOT / "Смещение - конечное -14 .xlsx"
CASES = (
    ("Кремс", "Кремас", "971bff83-163e-4aec-a847-c9b475fc15c4"),
    ("Шилова", "Шилова", "3d67cbf2-4b47-4a17-a698-25af5bac909a"),
    ("Сомова", "Сомова", "c24911e6-afa6-4363-849c-36b6d91ed8e3"),
    ("Залесская", "Залесская", "d5b1e6c3-254b-4bea-a0ab-8c11feb07a30"),
    ("Вегерин", "Вегерин", "181c673f-4e7c-477c-9f10-ec19ebb4d5c9"),
)
AXES = "xyz"


def _oof_block(path: Path) -> dict:
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    oof = report.get("nested_groupkfold_oof") or {}
    return {
        "per_target_mae_mm": oof.get("per_target_mae_mm") or {},
        "axis_mae_mm": oof.get("axis_mae_mm") or {},
        "z_avg_mae_mm": oof.get("z_avg_mae_mm"),
        "avg_mae_mm": oof.get("avg_mae_mm"),
        "z_shrink_targets": report.get("z_shrink_targets") or [],
        "feature_count": report.get("feature_count"),
    }


def _case_row(case_id: str, surname: str, supine: pd.DataFrame, lateral: pd.DataFrame, bank: dict) -> dict:
    case = ROOT / "data" / "cases" / case_id / "artifacts"
    base = json.loads((case / "base_features.json").read_text(encoding="utf-8"))
    raw_path = case / "extraction_raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {}
    merged = dict(base)
    for key, value in raw.items():
        if merged.get(key) is None and value is not None:
            merged[key] = value
    merged["full_name"] = surname
    attached = attach_predictors(
        pd.DataFrame([merged]),
        supine,
        lateral,
        bank.get("supine_stats") or {},
        bank.get("lateral_stats") or {},
    )
    row = attached.iloc[0].to_dict()
    for key, value in merged.items():
        if key not in row:
            row[key] = value
            continue
        current = row.get(key)
        try:
            missing = bool(pd.isna(current))
        except (TypeError, ValueError):
            missing = False
        if missing:
            row[key] = value
    return row


def _predict(model: RuntimePredictor, row: dict) -> dict:
    raw = model.predict_row(row)
    predictions, sanity_ok, warnings, _flags, withheld = finalize_case_predictions(
        row=row,
        predictions=raw,
        payload=model.payload,
    )
    return {"predictions": predictions, "sanity_ok": sanity_ok, "warnings": warnings, "withheld": withheld}


def _truth_table() -> pd.DataFrame:
    sheet = parse_xlsx_raw_table(XLSX, exclude_surnames=[])
    sheet = sheet.copy()
    sheet["surname"] = sheet["fio"].map(normalize_surname_key)
    return sheet


def main() -> int:
    oof = {"current": _oof_block(OLD_REPORT), "ct_bank": _oof_block(NEW_REPORT)}
    print("[oof current]", json.dumps(oof["current"].get("axis_mae_mm"), ensure_ascii=False))
    print("[oof ct_bank]", json.dumps(oof["ct_bank"].get("axis_mae_mm"), ensure_ascii=False))
    print("[shrink current]", oof["current"].get("z_shrink_targets"), "features", oof["current"].get("feature_count"))
    print("[shrink ct_bank]", oof["ct_bank"].get("z_shrink_targets"), "features", oof["ct_bank"].get("feature_count"))

    bank = load_bank()
    supine = load_profile(ROOT / "na_spine_full_profile.csv")
    lateral = load_profile(ROOT / "na_boku_full_profile.csv")
    current = RuntimePredictor.load(OLD_MODEL)
    trained = RuntimePredictor.load(NEW_MODEL)
    sheet = _truth_table()
    rows = []
    for label, sheet_name, case_id in CASES:
        row = _case_row(case_id, sheet_name, supine, lateral, bank)
        old = _predict(current, row)
        new = _predict(trained, row)
        excel = sheet[sheet["surname"] == normalize_surname_key(sheet_name)]
        truth = excel.iloc[0] if not excel.empty else None
        for side in ("left", "right"):
            for axis in AXES:
                target = f"kidney_{side}_delta_{axis}"
                excel_value = None if truth is None else truth.get(f"{side}_delta_middle_{axis}")
                rows.append(
                    {
                        "surname": label,
                        "target": target,
                        "truth": None if excel_value is None or pd.isna(excel_value) else float(excel_value),
                        "current": old["predictions"].get(target),
                        "ct_bank": new["predictions"].get(target),
                    }
                )
        print(label, "current", old["predictions"])
        print(label, "ct_bank", new["predictions"])
    frame = pd.DataFrame(rows)
    use = frame.dropna(subset=["truth", "current", "ct_bank"]).copy()
    summary = {}
    for axis in AXES:
        part = use[use["target"].str.endswith(axis)]
        summary[axis] = {
            "n": int(len(part)),
            "current_mae": float((part["current"] - part["truth"]).abs().mean()) if len(part) else None,
            "current_bias": float((part["current"] - part["truth"]).mean()) if len(part) else None,
            "ct_bank_mae": float((part["ct_bank"] - part["truth"]).abs().mean()) if len(part) else None,
            "ct_bank_bias": float((part["ct_bank"] - part["truth"]).mean()) if len(part) else None,
        }
    print("[holdout]", json.dumps(summary, ensure_ascii=False, indent=2))
    out = ROOT / "results" / "ct_bank_model_compare.json"
    out.write_text(
        json.dumps({"oof": oof, "holdout_axis": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[ok]", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
