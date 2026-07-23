#!/usr/bin/env bash
set -euo pipefail

# Unified entrypoint for WSL validation.
# Usage:
#   bash scripts/validation/run_all.sh RUN_ID [MODE]
# MODE: all | visuals | metrics (default: all)

RUN_ID="${1:-}"
MODE="${2:-all}"

if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: bash scripts/validation/run_all.sh RUN_ID [all|visuals|metrics]"
  exit 2
fi

DATASET_PATH="${DATASET_PATH:-data/vybor_unified_features.csv}"
MODEL_PATH="${MODEL_PATH:-models/adaptive_ensemble_clinical_honest.pkl}"
OUT_DIR="${OUT_DIR:-results/validation_runs}"
NUM_CASES="${NUM_CASES:-8}"
SEED="${SEED:-42}"
TEST_SIZE="${TEST_SIZE:-0.3}"

echo "[INFO] Run id: ${RUN_ID}"
echo "[INFO] Mode: ${MODE}"
echo "[INFO] Dataset: ${DATASET_PATH}"
echo "[INFO] Model: ${MODEL_PATH}"

python3 scripts/validation/smoke_check.py \
  --dataset "${DATASET_PATH}" \
  --model "${MODEL_PATH}"

if [[ "${MODE}" == "all" || "${MODE}" == "visuals" ]]; then
  python3 scripts/validation/run_visual_tests.py \
    --dataset "${DATASET_PATH}" \
    --model "${MODEL_PATH}" \
    --run-id "${RUN_ID}" \
    --out-dir "${OUT_DIR}" \
    --num-cases "${NUM_CASES}" \
    --seed "${SEED}" \
    --test-size "${TEST_SIZE}"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "metrics" ]]; then
  python3 scripts/validation/evaluate_metrics.py \
    --dataset "${DATASET_PATH}" \
    --model "${MODEL_PATH}" \
    --run-id "${RUN_ID}" \
    --out-dir "${OUT_DIR}" \
    --seed "${SEED}" \
    --test-size "${TEST_SIZE}" \
    --top-n 10
fi

echo "[OK] Validation run completed: ${OUT_DIR}/${RUN_ID}"
