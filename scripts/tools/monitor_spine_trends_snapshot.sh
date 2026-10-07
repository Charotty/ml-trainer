#!/usr/bin/env bash
# Snapshot for SPINE progress monitor (CSV rows/ts_ok/mtime + pgrep + GPU)
set +e
LOG="/mnt/e/ml/ml-trainer/results/wsl_na_spine_finish.log"
CSV="/mnt/e/ml/ml-trainer/results/na_spine_full.csv"

echo "[procs]"
pgrep -af 'extract_from_dicom|wsl_finish_na_spine' || echo "  (none)"
echo
if [[ -f "$LOG" ]]; then
  echo "[log] $(wc -l < "$LOG") lines; mtime=$(date -r "$LOG" '+%F %T' 2>/dev/null || stat -c %y "$LOG" 2>/dev/null | cut -d. -f1); last 12:"
  tail -n 12 "$LOG"
else
  echo "[log] missing: $LOG"
fi
echo
python3 - <<'PY'
import csv
from datetime import datetime
from pathlib import Path

p = Path("/mnt/e/ml/ml-trainer/results/na_spine_full.csv")
if not p.exists():
    print("[csv] missing")
else:
    rows = extracted = ts_ok = 0
    with p.open(encoding="utf-8", newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            rows += 1
            if (row.get("status") or "").strip() == "extracted":
                extracted += 1
            if (row.get("totalsegmentator_status") or "").strip() == "ok":
                ts_ok += 1
    mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[csv] rows={rows} extracted={extracted} ts_ok={ts_ok} mtime={mtime}")
PY
echo
echo "[gpu]"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | sed 's/^/  /'
else
  echo "  nvidia-smi not available"
fi
