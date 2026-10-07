#!/usr/bin/env bash
# Stage fresh extracts -> data/, harmonize, retrain clinical honest (na_trends).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LOG_DIR="/tmp/ml_trainer_logs"
LOG="$LOG_DIR/wsl_retrain_clinical_honest.log"
mkdir -p "$LOG_DIR" results
: > "$LOG"
exec > >(tee "$LOG") 2>&1

export PYTHONUNBUFFERED=1

echo "============================================================"
echo "[start] $(date -Is)"
echo "[cwd]   $ROOT"
echo "[log]   $LOG"
echo "============================================================"

source /home/user/venv-ml-trainer/bin/activate
echo "[python] $(python -V) @ $(which python)"

echo
echo "[stage] copy results extracts -> data/"
cp -f results/na_spine_full.csv data/na_spine_full.csv
cp -f results/na_boku_full.csv data/na_boku_full.bak.csv
ls -la data/na_spine_full.csv data/na_boku_full.bak.csv data/vybor_from_xlsx.csv

echo
echo "[harmonize] rebuild data/harmonized from fresh extracts + vybor reference"
python scripts/data/harmonize_extracted_datasets.py \
  --spine data/na_spine_full.csv \
  --boku data/na_boku_full.bak.csv \
  --reference data/vybor_from_xlsx.csv

echo
echo "[train] clinical honest na_trends (skip vybor rebuild; no KiTS)"
python scripts/data/train_clinical_honest.py \
  --z-head ensemble \
  --skip-vybor-build \
  --vybor-csv data/vybor_from_xlsx.csv \
  --spine-csv data/harmonized/na_spine_full_aligned.csv \
  --boku-csv data/harmonized/na_boku_full_aligned.csv \
  --model-path models/adaptive_ensemble_clinical_honest.pkl

echo
echo "[done] $(date -Is)"
echo "DONE_MARKER_OK"
