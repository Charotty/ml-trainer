# Repo work checklist (honest clinical path)

Operational path from raw `dicexe/` sources to a running API. Work on branch `feature/ct-workbench-ui-spec`. Canonical production model: `models/adaptive_ensemble_clinical_honest.pkl` (Adaptive Ensemble, ~121 features including clinical demographics + `na_trends`; GKF-OOF on n≈87). KiTS is optional and not required for this path.

Related: [VALIDATION_READINESS_CHECKLIST.md](VALIDATION_READINESS_CHECKLIST.md) (post-run validation gates).

---

## 0. Branch and raw sources

- [ ] `git rev-parse --abbrev-ref HEAD` → `feature/ct-workbench-ui-spec`
- [ ] Raw inputs present under `dicexe/`:
  - [ ] `dicexe/na_spine_full.csv`
  - [ ] `dicexe/na_boku_full.csv`
  - [ ] displacement workbook, e.g. `dicexe/Смещение - конечное -12 .xlsx`

---

## 1. Staging (`dicexe/` → `data/`)

Code reads `data/`, not `dicexe/` directly. CSV/XLSX are usually gitignored.

| From `dicexe/` | To code path |
|----------------|--------------|
| `na_spine_full.csv` | `data/na_spine_full.csv` |
| `na_boku_full.csv` | `data/na_boku_full.bak.csv` (note `.bak`) |
| displacement xlsx | `data/*.xlsx` (or pass `--xlsx` to build) |

- [ ] `Test-Path data/na_spine_full.csv`
- [ ] `Test-Path data/na_boku_full.bak.csv`
- [ ] xlsx in `data/` **or** documented `--xlsx` path to `dicexe/`

KiTS (`kits19_*`) is **not** staged from `dicexe` and is not required for honest training.

---

## 2. Build clinical labels CSV

```powershell
py -3 scripts/data/build_vybor_from_xlsx.py --no-boku
# if no xlsx under data/:
# py -3 scripts/data/build_vybor_from_xlsx.py --no-boku --xlsx "dicexe/Смещение - конечное -12 .xlsx"
```

- [ ] `data/vybor_from_xlsx.csv` exists
- [ ] All 6 targets present: `kidney_left_delta_x/y/z`, `kidney_right_delta_x/y/z`
- [ ] Manifest written: `data/vybor_from_xlsx.manifest.json`

---

## 3. Train (honest)

```powershell
py -3 scripts/data/train_clinical_honest.py --z-head ensemble
# optional experiment only:
# py -3 scripts/data/train_clinical_honest.py --z-head ensemble --with-kits
```

- [ ] Output: `models/adaptive_ensemble_clinical_honest.pkl`
- [ ] Bundle includes enrichment / trend store as produced by the trainer

---

## 4. Validate / smoke

```powershell
py -3 scripts/validation/smoke_check.py
# then full validation run per VALIDATION_READINESS_CHECKLIST.md / WSL runbook
```

- [ ] Smoke exits successfully with the clinical_honest model
- [ ] Validation readiness gates in [VALIDATION_READINESS_CHECKLIST.md](VALIDATION_READINESS_CHECKLIST.md)

---

## 5. API

```powershell
uvicorn src.api.kidney_displacement_api:app --host 127.0.0.1 --port 8000
# or: py -3 src/api/kidney_displacement_api.py
```

- [ ] API loads `models/adaptive_ensemble_clinical_honest.pkl` (canonical default)
- [ ] Predict path uses the same feature enrichment as training

---

## Notes

- **Honest vs proxy:** production labels come only from the displacement xlsx → `vybor_from_xlsx.csv`. `na_spine` / `na_boku` are for cohort trend features, not paired δ labels. Proxy trainers are research-only, not production.
- **KiTS:** optional for trend enrichment (`--with-kits`); skip for the default honest path.
- **Canonical code path:** `scripts/data/train_clinical_honest.py`, `scripts/validation/*`, `src/api/kidney_displacement_api.py`. Do not rely on deleted phase2/phase3 research trainers.
- Deeper data roles: [DATA_ARCHITECTURE.md](DATA_ARCHITECTURE.md). Command dump: [PIPELINE_COMMANDS.md](PIPELINE_COMMANDS.md).

---

## Definition of Done

Staging and honest path are complete when all of the following hold:

- [ ] Branch is `feature/ct-workbench-ui-spec`
- [ ] Staged inputs exist: `data/na_spine_full.csv`, `data/na_boku_full.bak.csv`, and xlsx (or `--xlsx`)
- [ ] `data/vybor_from_xlsx.csv` has all 6 delta targets with complete rows (e.g. 87/87)
- [ ] `models/adaptive_ensemble_clinical_honest.pkl` loads via `load_model_bundle` with `enrichment_mode`, `na_trend_store`, `imputer`/`scaler`/`models`, and Z calibrators as applicable
- [ ] Smoke / validation gates in [VALIDATION_READINESS_CHECKLIST.md](VALIDATION_READINESS_CHECKLIST.md) pass against clinical_honest (not legacy RF / proxy)
- [ ] API defaults to clinical_honest; proxy ≠ production; KiTS remains optional
- [ ] CT extractor does not invent synthetic deltas; seed/determinism is set where RNG remains
