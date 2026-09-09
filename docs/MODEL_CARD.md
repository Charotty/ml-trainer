# Model card index (metrics source of truth)

**This file plus the archive JSON cards and [`MODEL_RELEASE_GATES.md`](MODEL_RELEASE_GATES.md) are the only current sources for model metrics and promotion status.**

Do not cite README tables, `MODEL_TECHNICAL_DOCUMENTATION.md` MAE 2.14 mm, or historical GKF-OOF 8.40 / 8.49 / 8.52 mm as production accuracy.

## Freeze state (no winner)

| Candidate | Features | SHA-256 prefix | Source | `production_winner` |
|-----------|---------:|----------------|--------|---------------------|
| `adaptive_ensemble_clinical_honest_f111_3E03B8FA` | 111 | `3E03B8FA` | working-tree snapshot | **false** |
| `adaptive_ensemble_clinical_honest_f121_5F317838` | 121 | `5F317838` | Git HEAD archive | **false** |

Cards: `models/archive/*_f111_*.json` and `models/archive/*_f121_*.json`.

The mutable alias `models/adaptive_ensemble_clinical_honest.pkl` is **not** a release identifier. The working alias currently loads the **111-feature** file; the git-tracked archive binary is **121 features**. Both stay `research_only: true`.

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
