#!/usr/bin/env bash
set -euo pipefail
source ~/venv-ml-trainer/bin/activate
cd /mnt/e/ml/ml-trainer

BOKU=$(ls -d /mnt/f/На\ Боку/*/ 2>/dev/null | head -1)
SPINE_ROOT="/mnt/f/На спине"

echo "=== prep test boku: $BOKU ==="
python scripts/inference/extract_from_dicom.py "$BOKU" \
  --prep-only --max-cases 1 \
  --output results/test_prep_boku.csv \
  --temp-dir /tmp/ml_dicom_test

echo "=== discover counts ==="
python - <<'PY'
from pathlib import Path
from scripts.inference.dicom_prep import discover_patient_cases, count_dicom_files, group_dicom_series, select_main_ct_series

for label, root in [("boku", Path("/mnt/f/На Боку")), ("spine", Path("/mnt/f/На спине"))]:
    cases = discover_patient_cases(root)
    print(label, "cases", len(cases), "sample", cases[0].name if cases else None)
    if cases:
        sm = group_dicom_series(cases[0])
        main = select_main_ct_series(sm)
        print("  series", len(sm), "main", main.slice_count if main else None, main.description[:40] if main else "")
PY
