#!/usr/bin/env python3
"""Summarize extract_from_dicom / batch CSV quality."""

from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter
from pathlib import Path


def fnum(x: str | None) -> float | None:
    if x is None or str(x).strip() in ("", "nan", "None"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def nn(rows: list[dict], col: str) -> int:
    return sum(1 for r in rows if str(r.get(col, "")).strip() not in ("", "nan", "None"))


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/na_boku_full.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    n = len(rows)
    print(f"FILE: {path}")
    print(f"rows={n} cols={len(rows[0]) if rows else 0}\n")

    for col in ("status", "totalsegmentator_status", "kidney_source"):
        print(f"{col}: {dict(Counter(r.get(col, '') for r in rows))}")

    print("\n--- completeness ---")
    for col in (
        "kidney_left_volume_cm3",
        "kidney_right_volume_cm3",
        "kidney_left_center_x",
        "kidney_right_center_x",
        "body_volume_cm3",
        "spine_center_x",
        "age",
        "sex",
        "bmi",
        "weight_kg",
        "height_m",
        "slice_count_used",
        "series_slices",
    ):
        if rows and col in rows[0]:
            c = nn(rows, col)
            print(f"  {col}: {c}/{n} ({100 * c / n:.1f}%)")

    lv = [fnum(r.get("kidney_left_volume_cm3")) for r in rows]
    rv = [fnum(r.get("kidney_right_volume_cm3")) for r in rows]
    both = sum(1 for a, b in zip(lv, rv) if a is not None and b is not None)
    print(f"\n--- kidneys ---")
    print(f"  both: {both}, left_only: {sum(1 for a,b in zip(lv,rv) if a and not b)}")
    print(f"  right_only: {sum(1 for a,b in zip(lv,rv) if b and not a)}, neither: {sum(1 for a,b in zip(lv,rv) if not a and not b)}")

    for side, vals in (("left", lv), ("right", rv)):
        v = [x for x in vals if x is not None]
        if v:
            print(f"  kidney_{side}_volume_cm3: mean={statistics.mean(v):.1f} min={min(v):.1f} max={max(v):.1f}")

    ss = [fnum(r.get("series_slices")) for r in rows]
    ss_ok = [x for x in ss if x is not None]
    print(f"\n--- series_slices ---")
    print(f"  min={min(ss_ok):.0f} median={statistics.median(ss_ok):.0f} max={max(ss_ok):.0f}")
    print(f"  <500: {sum(1 for x in ss_ok if x < 500)}  <300: {sum(1 for x in ss_ok if x < 300)}")

  # prep warnings
    tags: Counter[str] = Counter()
    for r in rows:
        for part in (r.get("prep_warnings") or "").split(";"):
            if part.strip():
                tags[part.split("=")[0]] += 1
    print(f"\n--- prep_warnings tags ---")
    for k, v in tags.most_common():
        print(f"  {k}: {v}/{n}")

    errs = [r for r in rows if str(r.get("error", "")).strip()]
    print(f"\n--- errors ({len(errs)}) ---")
    for msg, cnt in Counter(r.get("error", "") for r in errs).most_common(8):
        print(f"  [{cnt}] {msg[:100]}")

    bad_ts = [r for r in rows if r.get("totalsegmentator_status") not in ("ok", "")]
    print(f"\n--- TS not ok ({len(bad_ts)}) ---")
    for r in bad_ts[:12]:
        print(
            f"  {r.get('case_id', '')[:50]} | {r.get('totalsegmentator_status')} | "
            f"slices={r.get('series_slices')} | {str(r.get('prep_warnings', ''))[:50]}"
        )

    print(f"\n--- demographics quirks ---")
    print(f"  missing weight_kg: {n - nn(rows, 'weight_kg')}")
    print(f"  missing height_m: {n - nn(rows, 'height_m')}")
    print(f"  bmi==25.0 (default?): {sum(1 for r in rows if r.get('bmi') == '25.0')}")
    print(f"  age present: {nn(rows, 'age')}")

    partial = [r for r in rows if r.get("status") not in ("extracted",)]
    print(f"\n--- status not 'extracted' ({len(partial)}) ---")
    for r in partial:
        print(f"  {r.get('case_id', '')[:50]} | {r.get('status')} | {str(r.get('error', ''))[:60]}")

    desc_ok = Counter(r.get("series_description", "").strip() for r in rows if r.get("totalsegmentator_status") == "ok")
    desc_fail = Counter(r.get("series_description", "").strip() for r in rows if r.get("totalsegmentator_status") == "failed")
    print("\n--- series_description (TS ok vs failed) ---")
    print("  ok:", desc_ok.most_common(4))
    print("  failed:", desc_fail.most_common(4))

    extracted_no_kidney = sum(
        1 for r in rows if r.get("status") == "extracted" and r.get("kidney_source") != "totalsegmentator_nifti"
    )
    print(f"\n--- extracted but no TS kidneys: {extracted_no_kidney}/{n} ---")
    print(f"  (body/spine metrics from enhanced_ct_extractor still present)")

    zero_left = sum(1 for r in rows if r.get("kidney_left_volume_cm3") in ("0.0", "0"))
    print(f"  kidney_left_volume_cm3 == 0: {zero_left}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
