#!/usr/bin/env python3
"""Compare CT Workbench predictions for Жильцова against the Excel GT row.

Reads the same xlsx parser as training, pulls features/prediction from the
Cases API (or local case artifacts), writes
``results/zhiltsova_step_<id>_vs_excel.csv`` and a MAE diff vs a baseline CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.excel_displacement_adapter import _body_type_to_code  # noqa: E402
from src.data.xlsx_displacement_parser import (  # noqa: E402
    DEFAULT_XLSX_PATH,
    build_vybor_from_xlsx,
    parse_xlsx_raw_table,
)
from src.features.laterality import (  # noqa: E402
    KIDNEY_LEFT_PRESENT,
    KIDNEY_RIGHT_PRESENT,
    laterality_from_excel_labels,
)
from src.features.phase1_schema import TARGET_NAMES  # noqa: E402

DEFAULT_CASE_ID = "3c2c292f-49d7-474b-86f5-6b74746c17dc"
DEFAULT_BASELINE = REPO_ROOT / "results" / "zhiltsova_excel_vs_web.csv"
DEFAULT_API = "http://127.0.0.1:8010"
DEFAULT_SURNAME = "жильцов"

EXCEL_CLINIC_FIELDS: Tuple[str, ...] = (
    "bmi",
    "has_previous_surgery",
    "body_type",
    "lumbar_lordosis_deg",
    "s1_plate_tilt_deg",
    "abd_wall_thickness_mm",
    "body_width_mm",
    "body_depth_mm",
    "kidney_left_z_span_supine_mm",
    "kidney_right_z_span_supine_mm",
    "kidney_left_y_span_supine_mm",
    "kidney_right_y_span_supine_mm",
)

COMPARE_ROWS: Tuple[Tuple[str, str, str], ...] = (
    ("kidney_left_delta_x", "mm", "Excel: middle-third lateral-supine; Web: model prediction"),
    ("kidney_left_delta_y", "mm", "Excel: middle-third lateral-supine; Web: model prediction"),
    ("kidney_left_delta_z", "mm", "Excel: middle-third lateral-supine; Web: model prediction"),
    ("kidney_right_delta_x", "mm", "Excel: middle-third lateral-supine; Web: model prediction"),
    ("kidney_right_delta_y", "mm", "Excel: middle-third lateral-supine; Web: model prediction"),
    ("kidney_right_delta_z", "mm", "Excel: middle-third lateral-supine; Web: model prediction"),
    ("age", "years", "DICOM tag vs Excel"),
    ("sex_female", "0/1", "Excel ж/м; Web DICOM sex=2 female"),
    ("bmi", "kg/m2", "clinical table / QA"),
    ("has_previous_surgery", "0/1", "clinical table / QA"),
    ("abd_wall_thickness_mm", "mm", "clinical table / QA"),
    ("lumbar_lordosis_deg", "deg", "clinical table / QA"),
    ("s1_plate_tilt_deg", "deg", "clinical table / QA"),
    ("body_width_mm", "mm", "Excel L3–L4 transverse; Web extract or QA"),
    ("body_depth_mm", "mm", "Excel L3–L4 AP; Web extract or QA"),
    ("kidney_right_volume_cm3", "cm3", "Excel col20; Web TotalSegmentator"),
    ("kidney_left_volume_cm3", "cm3", "Excel col62; Web TotalSegmentator"),
    ("kidney_left_z_span_supine_mm", "mm", "upper-lower |Z|"),
    ("kidney_right_z_span_supine_mm", "mm", "upper-lower |Z|"),
    ("kidney_left_y_span_supine_mm", "mm", "upper-lower |Y|"),
    ("kidney_right_y_span_supine_mm", "mm", "upper-lower |Y|"),
)


def _finite(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def resolve_xlsx(path: Optional[Path]) -> Path:
    if path is not None and path.exists():
        return path
    candidates = [
        REPO_ROOT / "Смещение - конечное -13 .xlsx",
        REPO_ROOT / "Смещение - конечное -12  (2).xlsx",
        DEFAULT_XLSX_PATH,
    ]
    candidates.extend(sorted(REPO_ROOT.glob("Смещение*.xlsx")))
    candidates.extend(sorted((REPO_ROOT / "data").glob("*.xlsx")))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError("Excel displacement workbook not found")


def load_excel_patient(xlsx: Path, surname: str = DEFAULT_SURNAME) -> Tuple[pd.Series, pd.Series]:
    raw = parse_xlsx_raw_table(xlsx)
    raw_mask = raw["fio"].astype(str).str.lower().str.contains(surname, na=False)
    if not raw_mask.any():
        raise SystemExit(f"Patient {surname!r} not in {xlsx}")
    raw_row = raw.loc[raw_mask].iloc[0]
    converted = build_vybor_from_xlsx(xlsx, boku_path=None)
    name_col = "full_name" if "full_name" in converted.columns else "fio"
    conv_mask = converted[name_col].astype(str).str.lower().str.contains(surname, na=False)
    if not conv_mask.any():
        raise SystemExit(f"Patient {surname!r} missing after convert in {xlsx}")
    return raw_row, converted.loc[conv_mask].iloc[0]


def excel_metric_map(raw: pd.Series, converted: pd.Series) -> Dict[str, Optional[float]]:
    sex = converted.get("sex")
    sex_female = None
    sex_num = _finite(sex)
    if sex_num is not None:
        sex_female = 1.0 if sex_num == 2.0 else 0.0
    elif str(raw.get("sex") or "").strip().lower().startswith("ж"):
        sex_female = 1.0
    out: Dict[str, Optional[float]] = {
        "age": _finite(converted.get("age", raw.get("age"))),
        "sex_female": sex_female,
        "bmi": _finite(converted.get("bmi", raw.get("bmi"))),
        "has_previous_surgery": _finite(
            converted.get("has_previous_surgery", raw.get("has_previous_surgery"))
        ),
        "abd_wall_thickness_mm": _finite(converted.get("abd_wall_thickness_mm")),
        "lumbar_lordosis_deg": _finite(converted.get("lumbar_lordosis_deg")),
        "s1_plate_tilt_deg": _finite(converted.get("s1_plate_tilt_deg")),
        "body_width_mm": _finite(converted.get("body_width_mm", raw.get("abd_width_l3l4_mm"))),
        "body_depth_mm": _finite(converted.get("body_depth_mm", raw.get("abd_depth_l3l4_mm"))),
        "kidney_left_volume_cm3": _finite(converted.get("kidney_left_volume_cm3")),
        "kidney_right_volume_cm3": _finite(converted.get("kidney_right_volume_cm3")),
    }
    for name in TARGET_NAMES:
        out[name] = _finite(converted.get(name, raw.get(name.replace("kidney_", "").replace("delta_", "delta_middle_"))))
        if out[name] is None:
            side, _, axis = name.replace("kidney_", "").split("_")
            out[name] = _finite(raw.get(f"{side}_delta_middle_{axis}"))
    for col in (
        "kidney_left_z_span_supine_mm",
        "kidney_right_z_span_supine_mm",
        "kidney_left_y_span_supine_mm",
        "kidney_right_y_span_supine_mm",
    ):
        out[col] = _finite(converted.get(col, raw.get(col)))
    return out


def excel_clinic_overrides(converted: pd.Series, raw: pd.Series) -> Dict[str, Any]:
    overrides: Dict[str, float] = {}
    for field in EXCEL_CLINIC_FIELDS:
        value: object
        if field == "body_type":
            raw_val = converted.get("body_type", raw.get("body_type"))
            coded = _body_type_to_code(raw_val)
            value = coded if _finite(coded) is not None else raw_val
        elif field == "body_width_mm":
            value = converted.get("body_width_mm", raw.get("abd_width_l3l4_mm"))
        elif field == "body_depth_mm":
            value = converted.get("body_depth_mm", raw.get("abd_depth_l3l4_mm"))
        else:
            value = converted.get(field, raw.get(field))
        number = _finite(value)
        if number is not None:
            overrides[field] = number
    width = overrides.get("body_width_mm")
    depth = overrides.get("body_depth_mm")
    if width is not None and depth is not None:
        overrides["body_area_mm2"] = width * depth
    flags = laterality_from_excel_labels(
        left_volume=converted.get("kidney_left_volume_cm3", raw.get("kidney_left_volume_cm3")),
        right_volume=converted.get("kidney_right_volume_cm3", raw.get("kidney_right_volume_cm3")),
        left_delta_x=converted.get("kidney_left_delta_x", raw.get("kidney_left_delta_x")),
        right_delta_x=converted.get("kidney_right_delta_x", raw.get("kidney_right_delta_x")),
    )
    overrides[KIDNEY_LEFT_PRESENT] = flags["left"]
    overrides[KIDNEY_RIGHT_PRESENT] = flags["right"]
    return overrides


def _json_request(
    url: str,
    *,
    method: str = "GET",
    payload: Optional[Mapping[str, Any]] = None,
    timeout: float = 120.0,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{method} {url} -> {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(str(exc.reason if getattr(exc, "reason", None) else exc)) from exc
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def inprocess_request(
    client: Any,
    path: str,
    *,
    method: str = "GET",
    payload: Optional[Mapping[str, Any]] = None,
) -> Any:
    if method == "GET":
        response = client.get(path)
    elif method == "POST":
        response = client.post(path, json=payload)
    elif method == "PATCH":
        response = client.patch(path, json=payload)
    else:
        raise ValueError(method)
    if response.status_code >= 400:
        raise SystemExit(f"{method} {path} -> {response.status_code}: {response.text}")
    return response.json()


def make_inprocess_client():
    from fastapi.testclient import TestClient

    from src.api.cases.predictor import ProductionPredictor
    from src.api.cases.storage import CaseStorage
    from src.api.ct_workbench_api import create_app

    predictor = ProductionPredictor.load()
    storage = CaseStorage()
    application = create_app(storage=storage, predictor_factory=lambda: predictor)
    return TestClient(application)


def load_local_case(case_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    artifacts = REPO_ROOT / "data" / "cases" / case_id / "artifacts"
    features = json.loads((artifacts / "features.json").read_text(encoding="utf-8"))
    base = json.loads((artifacts / "base_features.json").read_text(encoding="utf-8"))
    prediction_path = artifacts / "prediction.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8")) if prediction_path.exists() else {}
    feat_doc = {
        "base_features": base,
        "all_features": features.get("all_features", {}),
        "coverage_pct": features.get("coverage_pct"),
        "missing_features": features.get("missing_features", []),
    }
    return feat_doc, prediction, {"source": "local_artifacts", "coverage_pct": feat_doc["coverage_pct"]}


def web_metric_map(feat_doc: Mapping[str, Any], prediction: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    base = dict(feat_doc.get("base_features") or {})
    all_features = dict(feat_doc.get("all_features") or {})
    merged = {**all_features, **base}
    preds = dict(prediction.get("predictions") or prediction)
    sex = _finite(merged.get("sex"))
    sex_female = 1.0 if sex == 2.0 else (0.0 if sex == 1.0 else None)
    out: Dict[str, Optional[float]] = {
        "age": _finite(merged.get("age")),
        "sex_female": sex_female,
        "bmi": _finite(merged.get("bmi")),
        "has_previous_surgery": _finite(merged.get("has_previous_surgery")),
        "abd_wall_thickness_mm": _finite(merged.get("abd_wall_thickness_mm")),
        "lumbar_lordosis_deg": _finite(merged.get("lumbar_lordosis_deg")),
        "s1_plate_tilt_deg": _finite(merged.get("s1_plate_tilt_deg")),
        "body_width_mm": _finite(merged.get("body_width_mm")),
        "body_depth_mm": _finite(merged.get("body_depth_mm")),
        "kidney_left_volume_cm3": _finite(merged.get("kidney_left_volume_cm3")),
        "kidney_right_volume_cm3": _finite(merged.get("kidney_right_volume_cm3")),
        "kidney_left_z_span_supine_mm": _finite(merged.get("kidney_left_z_span_supine_mm")),
        "kidney_right_z_span_supine_mm": _finite(merged.get("kidney_right_z_span_supine_mm")),
        "kidney_left_y_span_supine_mm": _finite(merged.get("kidney_left_y_span_supine_mm")),
        "kidney_right_y_span_supine_mm": _finite(merged.get("kidney_right_y_span_supine_mm")),
    }
    for name in TARGET_NAMES:
        out[name] = _finite(preds.get(name))
    return out


def comparison_rows(
    excel_map: Mapping[str, Optional[float]],
    web_map: Mapping[str, Optional[float]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric, unit, note in COMPARE_ROWS:
        excel = excel_map.get(metric)
        web = web_map.get(metric)
        delta = None
        if excel is not None and web is not None:
            delta = web - excel
        rows.append(
            {
                "metric": metric,
                "excel": excel,
                "web": web,
                "delta_web_minus_excel": delta,
                "unit": unit,
                "note": note,
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric", "excel", "web", "delta_web_minus_excel", "unit", "note"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def target_mae(rows: Iterable[Mapping[str, Any]]) -> Optional[float]:
    errors = []
    for row in rows:
        if row["metric"] not in TARGET_NAMES:
            continue
        delta = row.get("delta_web_minus_excel")
        if delta is None:
            continue
        errors.append(abs(float(delta)))
    if not errors:
        return None
    return float(np.mean(errors))


def print_report(
    rows: List[Dict[str, Any]],
    *,
    step_id: str,
    coverage: object,
    baseline_rows: Optional[List[Dict[str, Any]]] = None,
) -> None:
    mae = target_mae(rows)
    print(f"step={step_id}  coverage={coverage}  MAE_6d={mae:.3f} mm" if mae is not None else f"step={step_id}")
    print(f"{'metric':32} {'excel':>10} {'web':>10} {'d':>10}")
    for row in rows:
        if row["metric"] not in TARGET_NAMES:
            continue
        excel = row["excel"]
        web = row["web"]
        delta = row["delta_web_minus_excel"]
        excel_s = f"{excel:10.2f}" if excel is not None else f"{'n/a':>10}"
        web_s = f"{web:10.2f}" if web is not None else f"{'n/a':>10}"
        delta_s = f"{delta:10.2f}" if delta is not None else f"{'n/a':>10}"
        print(f"{row['metric']:32} {excel_s} {web_s} {delta_s}")
    if not baseline_rows:
        return
    base_web = {r["metric"]: _finite(r.get("web")) for r in baseline_rows}
    print("\nvs baseline web:")
    for row in rows:
        if row["metric"] not in TARGET_NAMES:
            continue
        prev = base_web.get(row["metric"])
        cur = _finite(row.get("web"))
        if prev is None or cur is None:
            continue
        print(f"  {row['metric']:32} {prev:8.2f} -> {cur:8.2f}  ({cur - prev:+.2f})")
    base_mae = target_mae(baseline_rows)
    if mae is not None and base_mae is not None:
        print(f"MAE {base_mae:.3f} -> {mae:.3f} ({mae - base_mae:+.3f} mm)")


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            parsed = dict(raw)
            for key in ("excel", "web", "delta_web_minus_excel"):
                parsed[key] = _finite(raw.get(key))
            rows.append(parsed)
        return rows


def _meta_timestamp(meta: Mapping[str, Any]) -> float:
    raw = str(meta.get("updated_at") or "")
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def wait_for_fresh_analyze(
    fetch_status,
    *,
    started_at: float,
    timeout_sec: float = 3600.0,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_sec
    last: Dict[str, Any] = {}
    seen_extracting = False
    while time.time() < deadline:
        try:
            last = fetch_status()
        except SystemExit as exc:
            print(f"  status poll retry: {exc}", flush=True)
            time.sleep(2)
            continue
        status = str(last.get("status") or "")
        print(
            f"  status={status} {last.get('progress_pct')}% {last.get('stage') or ''} {last.get('message') or ''}",
            flush=True,
        )
        if status == "extracting":
            seen_extracting = True
        if status == "failed" and (seen_extracting or _meta_timestamp(last) >= started_at - 1):
            return last
        fresh = _meta_timestamp(last) >= started_at - 1
        if status == "features_ready" and (seen_extracting or fresh):
            return last
        time.sleep(5)
    raise SystemExit(f"Timed out waiting for analyze; last={last}")


def poll_status(api: str, case_id: str, timeout_sec: float = 3600.0) -> Dict[str, Any]:
    started_at = time.time()
    return wait_for_fresh_analyze(
        lambda: _json_request(f"{api}/api/v1/cases/{case_id}/status"),
        started_at=started_at,
        timeout_sec=timeout_sec,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Жильцова Excel vs Workbench")
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--surname", default=DEFAULT_SURNAME, help="Excel FIO substring, e.g. жильцов / рублевск")
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--xlsx", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--local", action="store_true", help="Read artifacts from data/cases, skip API")
    parser.add_argument(
        "--inprocess",
        action="store_true",
        help="Use in-process TestClient + live CaseStorage (new code, no HTTP server)",
    )
    parser.add_argument("--patch-excel-clinic", action="store_true")
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Rebuild base/features from extraction_raw.json without re-running TotalSegmentator",
    )
    parser.add_argument("--predict", action="store_true", help="POST /predict even if prediction.json exists")
    parser.add_argument("--analyze", action="store_true", help="POST /analyze?fast=true and wait")
    parser.add_argument("--analyze-full", action="store_true", help="POST /analyze?fast=false and wait")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path (default results/zhiltsova_step_<id>_vs_excel.csv)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xlsx = resolve_xlsx(args.xlsx)
    surname = str(args.surname).strip().lower().replace("ё", "е")
    raw_row, converted = load_excel_patient(xlsx, surname=surname)
    excel_map = excel_metric_map(raw_row, converted)
    api = args.api.rstrip("/")
    feat_doc: Dict[str, Any]
    prediction: Dict[str, Any]
    coverage: object = None

    if args.rebuild_features:
        from src.api.cases.features_service import rebuild_features_from_extraction
        from src.api.cases.predictor import ProductionPredictor
        from src.api.cases.storage import CaseStorage

        print(f"Rebuilding features from extraction_raw.json for {args.case_id}")
        _, _, coverage_rebuilt, missing = rebuild_features_from_extraction(
            CaseStorage(), args.case_id, ProductionPredictor.load()
        )
        print(f"  coverage={coverage_rebuilt:.1f}% missing={len(missing)}")

    if args.local:
        feat_doc, prediction, meta = load_local_case(args.case_id)
        coverage = meta.get("coverage_pct")
        if args.patch_excel_clinic or args.predict or args.analyze_full or args.analyze:
            raise SystemExit("--local cannot PATCH/predict/analyze; use --inprocess or the API")
    elif args.inprocess:
        client = make_inprocess_client()
        if args.analyze or args.analyze_full:
            fast_q = "false" if args.analyze_full else "true"
            print(f"POST /api/v1/cases/{args.case_id}/analyze?fast={fast_q} (in-process)")
            started_at = time.time()
            inprocess_request(
                client,
                f"/api/v1/cases/{args.case_id}/analyze?fast={fast_q}",
                method="POST",
            )
            last = wait_for_fresh_analyze(
                lambda: inprocess_request(client, f"/api/v1/cases/{args.case_id}/status"),
                started_at=started_at,
            )
            if last.get("status") == "failed":
                raise SystemExit(f"Analyze failed: {last.get('error')}")
        if args.patch_excel_clinic:
            overrides = excel_clinic_overrides(converted, raw_row)
            print(f"PATCH clinic overrides: {sorted(overrides)}")
            feat_doc = inprocess_request(
                client,
                f"/api/v1/cases/{args.case_id}/features/manual",
                method="PATCH",
                payload={"overrides": overrides, "reason": f"excel clinic {args.step_id}"},
            )
        else:
            feat_doc = inprocess_request(client, f"/api/v1/cases/{args.case_id}/features")
        coverage = feat_doc.get("coverage_pct")
        if args.predict or args.patch_excel_clinic or args.analyze_full or args.analyze:
            prediction = inprocess_request(
                client,
                f"/api/v1/cases/{args.case_id}/predict",
                method="POST",
            )
        else:
            try:
                prediction = inprocess_request(client, f"/api/v1/cases/{args.case_id}/prediction")
            except SystemExit:
                prediction = inprocess_request(
                    client,
                    f"/api/v1/cases/{args.case_id}/predict",
                    method="POST",
                )
    else:
        if args.analyze or args.analyze_full:
            fast_q = "false" if args.analyze_full else "true"
            print(f"POST {api}/api/v1/cases/{args.case_id}/analyze?fast={fast_q}")
            started_at = time.time()
            _json_request(
                f"{api}/api/v1/cases/{args.case_id}/analyze?fast={fast_q}",
                method="POST",
                timeout=30.0,
            )
            status = wait_for_fresh_analyze(
                lambda: _json_request(f"{api}/api/v1/cases/{args.case_id}/status"),
                started_at=started_at,
            )
            if status.get("status") == "failed":
                raise SystemExit(f"Analyze failed: {status.get('error')}")
        if args.patch_excel_clinic:
            overrides = excel_clinic_overrides(converted, raw_row)
            print(f"PATCH clinic overrides: {sorted(overrides)}")
            feat_doc = _json_request(
                f"{api}/api/v1/cases/{args.case_id}/features/manual",
                method="PATCH",
                payload={"overrides": overrides, "reason": f"excel clinic {args.step_id}"},
            )
        else:
            feat_doc = _json_request(f"{api}/api/v1/cases/{args.case_id}/features")
        coverage = feat_doc.get("coverage_pct")
        if args.predict or args.patch_excel_clinic or args.analyze_full or args.analyze:
            prediction = _json_request(
                f"{api}/api/v1/cases/{args.case_id}/predict",
                method="POST",
            )
        else:
            try:
                prediction = _json_request(f"{api}/api/v1/cases/{args.case_id}/prediction")
            except SystemExit:
                prediction = _json_request(
                    f"{api}/api/v1/cases/{args.case_id}/predict",
                    method="POST",
                )

    web_map = web_metric_map(feat_doc, prediction)
    rows = comparison_rows(excel_map, web_map)
    stem = "zhiltsova" if "жильцов" in surname else (
        "rublevskaya" if "рублев" in surname else "patient"
    )
    output = args.output or (REPO_ROOT / "results" / f"{stem}_step_{args.step_id}_vs_excel.csv")
    write_csv(output, rows)
    baseline_rows = read_csv_rows(args.baseline) if args.baseline.exists() else None
    print_report(rows, step_id=args.step_id, coverage=coverage, baseline_rows=baseline_rows)
    extra = {
        "step_id": args.step_id,
        "case_id": args.case_id,
        "xlsx": str(xlsx),
        "coverage_pct": coverage,
        "geometry_ood_x": (feat_doc.get("base_features") or {}).get("geometry_ood_x"),
        "missing_features": feat_doc.get("missing_features"),
        "output": str(output),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
