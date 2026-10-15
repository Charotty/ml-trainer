# Z frame fix, holdout of 15.10.2026

Branch `fix/z-frame-align` against tag `baseline-xlsx14`. The five holdout CTs were analyzed again in the Workbench (`fast=true`, vertebral frame on) and predicted with the retrained `models/adaptive_ensemble_clinical_honest.pkl` (123 features).

## What changed

The fast Workbench path now measures `*_vert` from named vertebrae. Those middle points are shifted into the Excel sheet frame (unsigned X, Y sign check, Z from the sheet’s vertebral zero) before prediction. `body_width_mm` outside 200–500 mm is left empty. Cohort z-scores use the sheet itself, and `na_pop_shift_*` is not a feature. If supine Z or the kidney-to-spine distance leaves the training p1–p99 window, ΔZ is the training mean and the case is flagged.

Nested GroupKFold OOF: left ΔZ MAE 11.43 mm versus a constant 10.43 mm, so the left axis is served as the training mean (−8.73 mm). Right ΔZ MAE 10.26 mm versus a constant 10.39 mm, so the right axis stays the model.

## Holdout versus the sheet

Error is prediction minus the sheet (lateral middle minus supine middle). Nine kidneys; Krems has no left kidney.

| Axis | Previous MAE | Previous bias | This run MAE | This run bias |
|------|-------------:|--------------:|-------------:|--------------:|
| X | 7.61 | +5.22 | 5.35 | +3.57 |
| Y | 8.14 | +7.38 | 5.70 | +1.13 |
| Z | 24.84 | −23.53 | 16.01 | −11.12 |

A constant at the training mean would score about 12 mm MAE on these nine Z values. The served mix (left constant, right model, plus the geometry fallback) is 16 mm. The previous run was 25 mm with bias −24 mm.

Krems has only the right kidney, so the two-kidney frame is not applied and right ΔZ falls back to the mean. Shilova’s kidney-to-spine distance is just outside the training window, so both of her ΔZ values are the mean as well.

## Experiment C

`exp/ct-feature-retrain` trains on CT-extractor geometry with sheet labels (`--feature-source ct`). The supine profile joins the sheet on 4 surnames (138 CT rows, 125 sheet rows). Nested GroupKFold needs 20, so no `clinical_ct_features.pkl` was written. The lateral profile joins 116 surnames; it was not used, because the Workbench reads a supine CT.
