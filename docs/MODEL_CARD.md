# Model card index (metrics source of truth)

**This file plus the archive JSON cards and [`MODEL_RELEASE_GATES.md`](MODEL_RELEASE_GATES.md) are the only current sources for model metrics and promotion status.**

Do not cite README tables, `MODEL_TECHNICAL_DOCUMENTATION.md` MAE 2.14 mm, or historical GKF-OOF 8.40 / 8.49 / 8.52 mm as production accuracy.

## Freeze state (no winner)

| Candidate | Features | SHA-256 prefix | Source | `production_winner` |
|-----------|---------:|----------------|--------|---------------------|
| `adaptive_ensemble_clinical_honest_f111_3E03B8FA` | 111 | `3E03B8FA` | working-tree snapshot | **false** |
| `adaptive_ensemble_clinical_honest_f121_5F317838` | 121 | `5F317838` | Git HEAD archive | **false** |

Cards: `models/archive/*_f111_*.json` and `models/archive/*_f121_*.json`.

The mutable alias `models/adaptive_ensemble_clinical_honest.pkl` is **not** a release identifier. As of 2026-10-03 the working alias is the Variant A selection below. Older archive binaries stay `research_only: true`.

## Variant A working alias (2026-10-03)

Selected on repeated nested GroupKFold, seeds 0, 1 and 2. Training used `Смещение - конечное -13 .xlsx` with the five holdout patients removed (Вегерин, Залесская, Сомова, Шилова, Кремас). Config: `config/variant_a_selected.json`.

| | |
|---|---|
| Model | random forest, one model per axis, Y/Z sample-weight boost off |
| Patients | 120 (102 both kidneys, 18 one kidney) |
| Features | 129 |
| Repeated-CV MAE | 7.31 mm (95% CI 6.73–8.01) |
| MAE X / Y / Z | 5.28 / 6.44 / 10.20 mm |
| 3D mean error | 15.10 mm; 30% of kidneys within 10 mm |
| R² | 0.055 |
| Holdout trio (xlsx features) | 7.27 mm — Кремас, Сомова, Вегерин |
| Journal row | stage 2 `2_yz_off`, refit recorded as stage 9 |

This alias is the file the workbench loads. It is **not** `production_winner`. [`MODEL_RELEASE_GATES.md`](MODEL_RELEASE_GATES.md) is unchanged: promotion is still blocked. Z error stays about 10 mm; later Variant A stages did not beat this forest by the 0.2 mm rule.

## Metrics policy

- Historical GKF-OOF MAE **8.40 / 8.49 / 8.52 mm** were produced under mixed protocols. They **MUST NOT** be mixed, compared across candidates, or cited as current production evidence.
- MAE **2.14 mm** in older technical docs is a **historical / leaky** in-sample figure (not nested OOF on n=87). It is **not** production.
- Corrected nested GroupKFold OOF on n=87 for **both** f111 and f121 is still required before any promotion.

## Environments (do not freeze every transitive pin here)

Training, CPU inference, and GPU extraction are **separate** stacks:

| Role | Typical file | sklearn pin (as of freeze) |
|------|----------------|----------------------------|
| CPU inference / Workbench | `requirements.txt`, `requirements-docker.txt` | `scikit-learn>=1.0.0,<1.6` |
| GPU extraction (WSL) | `requirements-gpu-wsl.txt` | `scikit-learn==1.5.2` |
| Artifact cards | `models/archive/*.json` | recorded **1.9.0** |

That sklearn 1.9 (cards) vs `<1.6` (inference requirements) mismatch is a known load risk. Load **joblib only from trusted sources** and check sklearn/joblib compatibility before serving. Runtime warns when the installed sklearn major.minor differs from a version recorded on the payload.

## Promotion

Blocked. See **PROMOTION_BLOCKED** in [`MODEL_RELEASE_GATES.md`](MODEL_RELEASE_GATES.md).
