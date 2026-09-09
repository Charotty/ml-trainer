# Model release gates

Neither archived candidate (`f111` nor `f121`) is a production winner. The mutable alias `models/adaptive_ensemble_clinical_honest.pkl` is not a release identifier. Promote an artifact only after **all** gates below pass, then publish it under an immutable filename plus a validated manifest (`artifact_schema_version` 1.0.0).

Immutable local copies (binaries are provenance-only, not the git source of truth):

- `models/archive/adaptive_ensemble_clinical_honest_f121_5F317838.pkl` — Git HEAD, 121 features
- `models/archive/adaptive_ensemble_clinical_honest_f111_3E03B8FA.pkl` — working-tree snapshot, 111 features

Paired JSON cards live next to those files and **are** commit-friendly.

## Required gates

1. **No leakage.** Training features must exclude lateral-scan targets, post-treatment outcomes, and any join that injects validation-patient information. Leakage-safe column filters and honest clinical labels are mandatory. A candidate that fails this gate stays research-only.

2. **Train / inference feature match.** The persisted `feature_names` list must be identical at train and serve time (count, order, and names). Preprocessor (imputer/scaler), enrichment mode (`na_trends` vs none), and calibrator presence must match the manifest. A silent drop/add of demographics or lordosis extras is a hard fail.

3. **Corrected nested OOF.** Outer GroupKFold evaluates only. Inner folds tune ensemble weights, hyperparameters, and extra heads. Imputer, scaler, feature selection, and enrichment statistics are refit inside each outer train split. Historical GKF-OOF MAE values **8.40 / 8.49 / 8.52 mm MUST NOT be mixed**, compared across the two candidates, or cited as production evidence until this protocol exists per artifact.

4. **3D and tail metrics.** Primary metric is 3D endpoint error for left and right kidney separately. Report median, p90, p95, max, and shares of patients within 5 / 10 / 15 mm. Axis MAE/RMSE/R² are secondary. Do not promote on mean MAE alone.

5. **External clinical validation or research-only label.** Either run a held-out clinical/DICOM cohort that was not used for training or model selection, **or** ship the artifact explicitly marked `research_only: true` / `status: research-only`. Missing this gate means the alias must not be advertised as clinical production.

## Metrics source of truth

Canonical metrics and status live in the archive JSON cards (`models/archive/*_f111_*.json`, `models/archive/*_f121_*.json`) and this file. Index: [`MODEL_CARD.md`](MODEL_CARD.md). README, API `/model_info`, Docker `/health`, and the clinical PDF must not invent MAE or name a winner.

## Environments and joblib loading

Training, CPU inference, and GPU CT extraction are **separate** environments. Do not assume one lockfile covers all three.

- Inference / Docker CPU: `requirements.txt` and `requirements-docker.txt` pin `scikit-learn>=1.0.0,<1.6`.
- GPU extraction: `requirements-gpu-wsl.txt` pins `scikit-learn==1.5.2`.
- Archive cards recorded **sklearn 1.9.0** / joblib 1.5.3 — that **conflicts** with the inference `<1.6` pin. Unpickle or prediction drift is possible until a matching training image is used.

Load joblib `.pkl` **only from trusted sources** (this repo’s `models/archive` hashes or a signed build). Before serve, compare installed sklearn major.minor to the artifact’s recorded version when present (`RuntimePredictor` warns on mismatch). Full transitive freeze is optional; the sklearn/joblib pair is not.

## Promotion rule

After gates 1–5 pass for exactly one candidate, copy that binary to a new immutable id, keep the previous hash for rollback, and only then retarget the production alias. Until then both cards stay `production_winner: false`.

## Rollback

1. Identify the currently mounted alias `models/adaptive_ensemble_clinical_honest.pkl` (mutable; not an ID).
2. Restore the previous immutable archive file by SHA-256 prefix:
   - `models/archive/adaptive_ensemble_clinical_honest_f111_3E03B8FA.pkl`
   - `models/archive/adaptive_ensemble_clinical_honest_f121_5F317838.pkl`
3. Copy that exact bytes onto the alias path (do not retrain, do not overwrite archive copies).
4. Smoke after deploy: `GET /health` (Workbench `:8010` or kidney API `:8000`) must show `model_loaded: true` and `feature_count` of **whatever was loaded** (111 or 121). Then one `predict` on a known row. Health must not claim a winner.

## PROMOTION_BLOCKED / next step

**No production winner is designated.** Do not retarget the alias to “the” model.

Next step (not done in this block): run **corrected nested GroupKFold OOF on n=87** for **f111 vs f121** under the same protocol, pick at most one candidate that passes gates 1–5, publish it under an **immutable model ID**, then optionally point the alias at that ID. Until that comparison exists, both candidates remain archived research-only artifacts.
