# Data architecture (correct split)

## Main table — only source of displacement labels

| Source | Role |
|--------|------|
| **Excel / `vybor_from_xlsx.csv`** | ~87–100 paired clinical patients (supine + lateral). **Only** source of true ΔX/ΔY/ΔZ for ensemble training. |

```bash
py -3 scripts/data/build_vybor_from_xlsx.py --no-boku
py -3 scripts/data/train_clinical_honest.py --z-head ensemble
```

Output model: `models/adaptive_ensemble_clinical_honest.pkl`

## Auxiliary CT extracts — trends only (na_spine / na_boku)

| File | Scan | Used for |
|------|------|----------|
| `data/na_spine_full.csv` | supine («на спине») | Cohort trends: supine kidney/body value distributions |
| `data/na_boku_full.bak.csv` | lateral («на боку») | Cohort trends: lateral value distributions + population shift priors |

**Not used for:**
- per-patient `proj_lat_*` / `proj_sup_*` joins by name (removed from honest path)
- filling missing volumes from boku (`--no-boku` on honest build)
- regression targets (no paired δ in these files)

**Used for (`src/features/na_trend_features.py`):**
- `na_pop_shift_{side}_{axis}` — median(lateral cohort) − median(supine cohort)
- `na_sup_z_*` / `na_sup_pct_*` — where clinical supine anatomy sits vs na_spine population

## KiTS19 — optional

Can be skipped for production. If used at all: external audit / feature imputation only, **not** clinical displacement labels in honest mode.

## Evaluation

Trust **GKF-5 OOF on 87 clinical** (~8.5 mm), not holdout-18 alone (~2.7 mm on full-train honest model).
