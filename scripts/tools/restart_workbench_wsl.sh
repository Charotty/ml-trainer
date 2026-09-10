#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/e/ml/ml-trainer"
VENV="/home/user/venv-ml-trainer"
LOG="/tmp/ct_workbench_uvicorn.log"
cd "$ROOT"
pkill -f "uvicorn src.api.ct_workbench_api:app" || true
sleep 1
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"
nohup python -m uvicorn src.api.ct_workbench_api:app --host 0.0.0.0 --port 8010 >"$LOG" 2>&1 &
echo "started pid=$! log=$LOG"
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8010/health >/tmp/wb_health.json 2>/dev/null; then
    cat /tmp/wb_health.json
    echo
    exit 0
  fi
  sleep 1
done
echo "Workbench failed to become healthy:"
tail -n 40 "$LOG" || true
exit 1
