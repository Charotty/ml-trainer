#!/usr/bin/env python3
"""Side-by-side Excel comparison for the five-case lateral pilot.

Writes one table: each kidney third, the sheet value, the four Lyashchenko
variants, and the absolute error of each variant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.compare_ct_excel_displacement import (  # noqa: E402
    _ct_label,
    _excel_coord_records,
    match_tables,
)
from scripts.validation.validate_extractor_anatomy import read_manual_table, surname_key  # noqa: E402
from src.features.ct_anatomy.vertebral_axes import (  # noqa: E402
    ORIGINS,
    POINT_KINDS,
    THIRDS,
    Z_REFERENCES,
    score_variant_pairs,
    variant_column,
    z_reference_column,
)

# DICOM PatientName is Latin, the sheet is Cyrillic. These five folders were
# the ones linked for the pilot, so the sheet surname comes from the folder.
FOLDER_SURNAMES = {
    "01_abdurakhmanov": "Абдурахманов",
    "02_abramovich": "Абрамович",
    "03_alieva": "Алиева",
    "04_albrandt": "Альбрандт",
    "05_amurkova": "Амуркова",
}
XLSX = Path("/mnt/f/Смещение - конечное -13 .xlsx")
EXTRACT = REPO_ROOT / "results" / "pilot5_boku.csv"
OUT = REPO_ROOT / "results" / "pilot5_boku_vs_excel.xlsx"


def _num(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number


def main() -> int:
    if not EXTRACT.exists():
        print(f"missing extract: {EXTRACT}")
        return 2
    frame = pd.read_csv(EXTRACT)
    ct_rows = []
    for _, row in frame.iterrows():
        folder_name = FOLDER_SURNAMES.get(str(row.get("case_id")), _ct_label(row))
        item = {"label": folder_name, "dicom_name": _ct_label(row)}
        item.update(row.to_dict())
        ct_rows.append(item)
    excel_rows = _excel_coord_records(XLSX)
    pairs, skipped = match_tables(excel_rows, ct_rows)
    print(f"paired {len(pairs)} held out {len(skipped)}")

    detail = []
    for excel, ct in pairs:
        for side in ("left", "right"):
            for third in THIRDS:
                base = {
                    "patient": ct.get("label"),
                    "dicom_name": ct.get("dicom_name"),
                    "excel_name": excel.get("label"),
                    "side": side,
                    "third": third,
                    "vert_level": ct.get(f"kidney_{side}_{third}_vert_level"),
                    "status": ct.get("status"),
                    "totalsegmentator_status": ct.get("totalsegmentator_status"),
                }
                for axis in ("x", "y", "z"):
                    manual = _num(excel.get(f"{side}_lateral_{third}_{axis}"))
                    row = dict(base)
                    row["axis"] = axis
                    row["excel"] = manual
                    for origin in ORIGINS:
                        for kind in POINT_KINDS:
                            name = f"{origin}_{kind}"
                            auto = _num(ct.get(variant_column(side, third, axis, origin, kind)))
                            row[name] = auto
                            row[f"{name}_abs_err"] = abs(auto - manual) if auto == auto and manual == manual else float("nan")
                    if axis == "z":
                        for level in Z_REFERENCES:
                            auto = _num(ct.get(z_reference_column(side, third, level)))
                            row[f"z_from_{level.lower()}"] = auto
                            row[f"z_from_{level.lower()}_abs_err"] = (
                                abs(auto - manual) if auto == auto and manual == manual else float("nan")
                            )
                    detail.append(row)

    anatomy = read_manual_table(XLSX)
    by_name = {str(row["name_key"]): row.to_dict() for _, row in anatomy.iterrows()}
    aligned_names = (
        ("perirenal_dorsal_vert_mm", "perirenal_dorsal_mm"),
        ("perirenal_ventral_vert_mm", "perirenal_ventral_mm"),
        ("psoas_thickness_vert_mm", "psoas_thickness_mm"),
        ("medial_to_spine_vert_mm", "medial_to_spine_mm"),
    )
    for excel, ct in pairs:
        sheet = by_name.get(surname_key(excel.get("label")))
        if sheet is None:
            continue
        for side in ("left", "right"):
            for ct_suffix, excel_suffix in aligned_names:
                manual = _num(sheet.get(f"kidney_{side}_{excel_suffix}"))
                auto = _num(ct.get(f"kidney_{side}_{ct_suffix}"))
                detail.append(
                    {
                        "patient": ct.get("label"),
                        "dicom_name": ct.get("dicom_name"),
                        "excel_name": excel.get("label"),
                        "side": side,
                        "third": "middle_slice",
                        "axis": ct_suffix,
                        "excel": manual,
                        "aligned": auto,
                        "aligned_abs_err": abs(auto - manual) if auto == auto and manual == manual else float("nan"),
                        "status": ct.get("status"),
                    }
                )

    detail_frame = pd.DataFrame(detail)
    summary = pd.DataFrame(score_variant_pairs(pairs, "lateral"))
    with pd.ExcelWriter(OUT) as writer:
        detail_frame.to_excel(writer, sheet_name="measurements", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    middle = summary[(summary["third"] == "middle") & (summary["axis"].isin(["x", "y"])) & (summary["n"] > 0)]
    if not middle.empty:
        print(middle.groupby("variant")["mae"].mean().sort_values().to_string())
    z_rows = summary[(summary["axis"] == "z") & (summary["third"] == "middle") & (summary["n"] > 0)]
    if not z_rows.empty:
        print(z_rows.sort_values("mae")[["variant", "mae", "bias_ct_minus_excel", "n"]].to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
