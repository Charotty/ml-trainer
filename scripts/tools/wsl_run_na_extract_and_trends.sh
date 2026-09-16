#!/usr/bin/env bash
# Full extract of F:/На спине + F:/На Боку, then NaTrendStore stats.
# Live log lives on Linux /tmp (DrvFS truncate via bash : > is unsafe).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LOG_DIR_LINUX="/tmp/ml_trainer_logs"
LOG_LINUX="$LOG_DIR_LINUX/wsl_na_extract_and_trends.log"
LOG_REPO="$ROOT/results/wsl_na_extract_and_trends.log"
mkdir -p "$LOG_DIR_LINUX" "$ROOT/results"

export PYTHONUNBUFFERED=1
: > "$LOG_LINUX"
# Do NOT truncate DrvFS paths with bash (: >) — optional mirror via cp at end.
exec > >(tee "$LOG_LINUX") 2>&1

echo "============================================================"
echo "[start] $(date -Is)"
echo "[cwd]   $ROOT"
echo "[log]   $LOG_LINUX"
echo "[mirror] $LOG_REPO (copied at end)"
echo "============================================================"

VENV=""
for cand in /home/user/venv-ml-trainer .venv-wsl; do
  if [[ -f "$cand/bin/activate" ]]; then
    VENV="$cand"
    break
  fi
done
if [[ -z "$VENV" ]]; then
  echo "ERROR: no usable venv (tried /home/user/venv-ml-trainer, .venv-wsl)"
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[venv]   $VENV"
echo "[python] $(python -V) @ $(which python)"
python - <<'PY' || true
import torch
print(f"[torch] {getattr(torch, '__version__', '?')} cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[gpu]   {torch.cuda.get_device_name(0)}")
PY

echo
echo "[extract] both DICOM roots via run_extract_jobs.py"
python scripts/tools/run_extract_jobs.py --device auto
EXTRACT_RC=$?
echo "[extract] exit_code=$EXTRACT_RC"
if [[ $EXTRACT_RC -ne 0 ]]; then
  echo "ERROR: extraction failed"
  exit "$EXTRACT_RC"
fi

echo
echo "[stats] analyze_extract_csv for both outputs"
python scripts/tools/analyze_extract_csv.py results/na_spine_full.csv || true
echo
python scripts/tools/analyze_extract_csv.py results/na_boku_full.csv || true

echo
echo "[trends] NaTrendStore.fit on extracted CSVs"
python - <<'PY'
from pathlib import Path
import json
from src.features.na_trend_features import NaTrendStore

root = Path(".").resolve()
spine = root / "results" / "na_spine_full.csv"
boku = root / "results" / "na_boku_full.csv"
store = NaTrendStore.fit(spine_path=spine, boku_path=boku, include_kits=False)
desc = store.describe()
print(json.dumps(desc, indent=2, ensure_ascii=False, default=str))
print("\n--- population_shift (mm, lateral − supine medians) ---")
for k, v in sorted(store.population_shift.items()):
    print(f"  {k}: {v:.4f}" if v == v else f"  {k}: nan")
print(f"\nsupine_stats columns: {len(store.supine_stats)}")
print(f"lateral_stats columns: {len(store.lateral_stats)}")
print(f"trend_features: {len(store.trend_feature_names())}")
for col in (
    "kidney_left_center_x_rel",
    "kidney_left_center_y_rel",
    "kidney_left_center_z_rel",
    "kidney_right_center_x_rel",
    "kidney_right_center_y_rel",
    "kidney_right_center_z_rel",
    "kidney_left_volume_cm3",
    "kidney_right_volume_cm3",
):
    s = store.supine_stats.get(col)
    l = store.lateral_stats.get(col)
    if s or l:
        print(f"\n{col}:")
        if s:
            print(f"  spine  n={s['n']} median={s['median']:.3f} mad={s['mad']:.3f} q10={s['q10']:.3f} q90={s['q90']:.3f}")
        if l:
            print(f"  boku   n={l['n']} median={l['median']:.3f} mad={l['mad']:.3f} q10={l['q10']:.3f} q90={l['q90']:.3f}")
PY

cp -f "$LOG_LINUX" "$LOG_REPO" 2>/dev/null || true

echo
echo "[done] $(date -Is)"
echo "DONE_MARKER_OK"
