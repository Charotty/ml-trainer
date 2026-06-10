# WSL Runbook: Project Validation

> **Полный цикл Phase 1 (extract → integrate → train → validate → API):**  
> см. [`docs/PHASE1_PIPELINE_RUNBOOK.md`](docs/PHASE1_PIPELINE_RUNBOOK.md)

This runbook provides copy-paste commands for preparing environment, running visual tests, collecting metrics, and storing reports.

## 1) Open repo in WSL

```bash
cd /mnt/d/ml\ trainer
```

## 2) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Optional: make entrypoint executable

```bash
chmod +x scripts/validation/run_all.sh
```

## 4) Smoke check only

```bash
python3 scripts/validation/smoke_check.py \
  --dataset data/processed/validation.csv \
  --model models/adaptive_ensemble.pkl
```

Or use the orchestrator after integrate + train:

```bash
python3 scripts/run_phase1_pipeline.py validate --run-id wsl_smoke_001
```

## 5) Full validation run (visuals + metrics)

```bash
RUN_ID="wsl_run_$(date +%Y%m%d_%H%M%S)"
bash scripts/validation/run_all.sh "$RUN_ID" all
```

## 6) Visual tests only

```bash
RUN_ID="visuals_only_$(date +%Y%m%d_%H%M%S)"
bash scripts/validation/run_all.sh "$RUN_ID" visuals
```

## 7) Metrics only

```bash
RUN_ID="metrics_only_$(date +%Y%m%d_%H%M%S)"
bash scripts/validation/run_all.sh "$RUN_ID" metrics
```

## 8) Override defaults via environment variables

```bash
export DATASET_PATH="data/processed/validation.csv"
export MODEL_PATH="models/adaptive_ensemble.pkl"
export OUT_DIR="results/validation_runs"
export NUM_CASES=12
export SEED=42
export TEST_SIZE=0.3

RUN_ID="custom_$(date +%Y%m%d_%H%M%S)"
bash scripts/validation/run_all.sh "$RUN_ID" all
```

## 9) API smoke test

In terminal A:

```bash
uvicorn src.api.kidney_displacement_api:app --host 127.0.0.1 --port 8000
```

In terminal B:

```bash
curl -s http://127.0.0.1:8000/health
```

## 10) Expected artifacts per run

Each run creates:

- `results/validation_runs/<RUN_ID>/plots/*.png`
- `results/validation_runs/<RUN_ID>/metrics/metrics_summary.csv`
- `results/validation_runs/<RUN_ID>/metrics/metrics_per_target.csv`
- `results/validation_runs/<RUN_ID>/metrics/worst_cases.csv`
- `results/validation_runs/<RUN_ID>/predictions/*.json`
- `results/validation_runs/<RUN_ID>/predictions/evaluation_predictions.csv`
- `results/validation_runs/<RUN_ID>/run_manifest.json`

## 11) Notes about model fallback

If `models/adaptive_ensemble.pkl` is missing, scripts automatically use a fallback RandomForest baseline trained on the provided dataset split. This lets you test pipeline reproducibility before restoring production model artifacts.

After `integrate` + `train`, validation uses the full 51-feature pipeline (normalize → engineering → imputer → scaler). Check `run_manifest.json`: `predictor_mode` must be `pretrained_adaptive_ensemble`.
