# Validation Readiness Checklist

Use this checklist after each WSL run.

For the full operational path (staging → build → train → validate → API), see [REPO_WORK_CHECKLIST.md](REPO_WORK_CHECKLIST.md).

## Technical readiness

- [ ] WSL environment starts and enters `/mnt/d/ml\ trainer`.
- [ ] Virtual environment is activated and `pip install -r requirements.txt` completes.
- [ ] `python3 scripts/validation/smoke_check.py` exits successfully.
- [ ] Validation command exits with status code `0`.

## Analytical readiness

- [ ] `metrics_per_target.csv` exists and contains all 6 targets.
- [ ] `metrics_summary.csv` includes `mae_avg_mm`, `rmse_avg_mm`, `r2_avg`, `within_5mm_ratio`, `within_10mm_ratio` (per-patient `vector_error_mean_mm`, not pointwise across 6 axes). Optional diagnostics: `within_5mm_pointwise_ratio`, `within_10mm_pointwise_ratio`.
- [ ] `worst_cases.csv` is generated and sorted by error descending.
- [ ] At least one visualization exists for each mode:
  - [ ] `single_case_3d`
  - [ ] `multi_panel_2d3d`
  - [ ] `overlay_supine_vs_predicted`

## Visual quality gates

- [ ] Left/right kidneys are not swapped.
- [ ] Axis labels match orientation: `X L->R`, `Y P->A`, `Z I->S`.
- [ ] 2D projections are directionally consistent with 3D view.
- [ ] Vertebra reference point is visible and stable in plots.

## Operational readiness

- [ ] All artifacts are inside `results/validation_runs/<RUN_ID>/`.
- [ ] `run_manifest.json` contains dataset/model paths and predictor mode.
- [ ] Rerun with another `RUN_ID` does not overwrite previous results.
- [ ] `WSL_RUNBOOK.md` commands reproduce the same process end-to-end.
