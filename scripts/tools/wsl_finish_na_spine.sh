#!/usr/bin/env bash
# Finish F:/На спине only (--update-existing), then NaTrendStore stats.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# IMPORTANT: creating new files under /mnt/e (DrvFS) from WSL often fails with
# "No such file or directory". Keep the live log on Linux /tmp and copy to
# results/ at the end (and optionally if Windows pre-created the mirror).
LOG_DIR_LINUX="/tmp/ml_trainer_logs"
LOG_LINUX="$LOG_DIR_LINUX/wsl_na_spine_finish.log"
LOG_REPO="$ROOT/results/wsl_na_spine_finish.log"
mkdir -p "$LOG_DIR_LINUX" "$ROOT/results"

export PYTHONUNBUFFERED=1

: > "$LOG_LINUX"

# If Windows already created the repo mirror, truncate it; never fail if not.
if [[ -f "$LOG_REPO" ]]; then
  : > "$LOG_REPO" || true
fi

if [[ -f "$LOG_REPO" ]]; then
  exec > >(tee "$LOG_LINUX" "$LOG_REPO") 2>&1
else
  exec > >(tee "$LOG_LINUX") 2>&1
fi

echo "============================================================"
echo "[start] $(date -Is)"
echo "[cwd]   $ROOT"
echo "[log]   $LOG_LINUX"
echo "[mirror] $LOG_REPO (optional DrvFS)"
echo "============================================================"

VENV="/home/user/venv-ml-trainer"
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "ERROR: missing $VENV"
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

SPINE_ROOT="/mnt/f/На спине"
SPINE_OUT="$ROOT/results/na_spine_full.csv"

echo
echo "[extract] spine only (dirs+zips, update-existing) -> $SPINE_OUT"
python scripts/inference/extract_from_dicom.py \
  --add-job "$SPINE_ROOT" "$SPINE_OUT" \
  --canonical \
  --device auto \
  --temp-dir /tmp/ml_trainer_dicom \
  --update-existing

echo
echo "[stats] analyze_extract_csv spine"
python scripts/tools/analyze_extract_csv.py results/na_spine_full.csv || true

echo
echo "[trends] NaTrendStore.fit (spine + boku)"
python - <<'PY'
from pathlib import Path
import json
from src.features.na_trend_features import NaTrendStore

root = Path(".").resolve()
store = NaTrendStore.fit(
    spine_path=root / "results" / "na_spine_full.csv",
    boku_path=root / "results" / "na_boku_full.csv",
    include_kits=False,
)
print(json.dumps(store.describe(), indent=2, ensure_ascii=False, default=str))
print("\n--- population_shift (mm, lateral - supine medians) ---")
for k, v in sorted(store.population_shift.items()):
    print(f"  {k}: {v:.4f}" if v == v else f"  {k}: nan")
print(f"\nsupine_stats={len(store.supine_stats)} lateral_stats={len(store.lateral_stats)}")
print(f"trend_features={len(store.trend_feature_names())}")
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
    if not (s or l):
        continue
    print(f"\n{col}:")
    if s:
        print(f"  spine  n={s['n']} median={s['median']:.3f} mad={s['mad']:.3f}")
    if l:
        print(f"  boku   n={l['n']} median={l['median']:.3f} mad={l['mad']:.3f}")
PY

# Best-effort copy to results for Windows browsing
cp -f "$LOG_LINUX" "$LOG_REPO" 2>/dev/null || true

echo
echo "[done] $(date -Is)"
echo "DONE_MARKER_OK"
