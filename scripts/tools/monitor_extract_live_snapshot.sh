#!/usr/bin/env bash
# Fallback live monitor snapshot for DICOM extract (read-only; does not touch job).
set +e
echo "ПРОЦЕССЫ / PROCESSES"
echo "  wsl_run_na_extract_and_trends.sh : $(kill -0 822 2>/dev/null && echo ALIVE || echo DEAD) (pid 822)"
echo "  tee                              : $(kill -0 826 2>/dev/null && echo ALIVE || echo DEAD) (pid 826)"
echo "  run_extract_jobs.py              : $(kill -0 850 2>/dev/null && echo ALIVE || echo DEAD) (pid 850)"
echo "  extract_from_dicom.py            : $(kill -0 851 2>/dev/null && echo ALIVE || echo DEAD) (pid 851)"
echo ""
echo "CSV ПРОГРЕСС / PROGRESS"
python3 - <<'PY'
from pathlib import Path
import csv
from datetime import datetime

def mtime(p: Path):
    try:
        return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "n/a"

def summarize(label, path):
    p = Path(path)
    if not p.exists():
        print(f"  {label}: MISSING {path}")
        return
    rows = extracted = ts_ok = 0
    try:
        with p.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows += 1
                if (row.get("status") or "").strip() == "extracted":
                    extracted += 1
                if (row.get("totalsegmentator_status") or "").strip() == "ok":
                    ts_ok += 1
    except Exception as e:
        print(f"  {label}: read error: {e}")
        return
    print(f"  {label}: rows={rows}  extracted={extracted}  ts_ok={ts_ok}  mtime={mtime(p)}")

summarize("na_boku_full.csv ", "/mnt/e/ml/ml-trainer/results/na_boku_full.csv")
summarize("na_spine_full.csv", "/mnt/e/ml/ml-trainer/results/na_spine_full.csv")
PY
echo ""
echo "TEMP /tmp/ml_trainer_dicom"
if [ -d /tmp/ml_trainer_dicom ]; then
  ls -1 /tmp/ml_trainer_dicom 2>/dev/null | sed "s/^/  /" | head -20
  n=$(ls -1 /tmp/ml_trainer_dicom 2>/dev/null | wc -l)
  echo "  (entries: $n)"
else
  echo "  (dir missing)"
fi
echo ""
echo "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | sed "s/^/  /"
else
  echo "  nvidia-smi not available"
fi
echo ""
if ! kill -0 851 2>/dev/null; then
  echo "*** extract_from_dicom.py no longer alive — job finished or stopped ***"
fi
