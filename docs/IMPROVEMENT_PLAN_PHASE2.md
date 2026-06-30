# План улучшения Phase 2 (displacement Y/Z)

## Ограничения

- Новых клинических экстремумов (|Δz|>20 mm) не будет — фокус на модели и aux-данных.
- `data/na_boku_full.bak.csv` — **непарная** таблица признаков в проекции **на боку** (lateral).
- `data/na_spine_full.csv` — **непарная** таблица признаков в проекции **на спине** (supine).
- Пары supine/lateral с δ берутся **только** из xlsx Vybor (87).

## Этапы

| # | Задача | Статус | Артефакты |
|---|--------|--------|-----------|
| 1 | Проекционное обогащение (boku/spine по `full_name_key`, без δ) | done | `src/features/projection_enrichment.py` |
| 2 | KiTS19 только для imputation + сниженный вес DICOM | done | `clinical_xlsx_kits_impute_only` |
| — | **COMMIT + PUSH** (чекпоинт перед 3–5) | done | `61c6505` |
| 3 | Отдельные модели `kidney_left_delta_z` / `kidney_right_delta_z` | done | `src/models/side_z_predictor.py` |
| 4 | Multitask (shared trunk + головы X/Y/Z) | done | `models/phase1/multitask_displacement.py` |
| 5 | Quantile prediction P10/P50/P90 | done | `src/models/quantile_displacement.py` |
| 6 | Скрипт обучения + валидация holdout | done | `scripts/data/train_phase2_improved.py` |

## Веса обучения (режим `clinical_xlsx_kits_impute_only`)

| Источник | В train для δ | sample_weight |
|----------|---------------|---------------|
| Vybor (xlsx) | да | 1.0 |
| DICOM pseudo | да | 0.06 |
| KiTS19 | **нет** (только median imputation) | — |

## Метрики успеха

- Holdout 18: MAE по L Δz, R Δz, среднее Z.
- Worst-case vector error (Алиева, Буров и др.).
- Ширина quantile-интервала на экстремумах (coverage 80%).

## Команды

```bash
py -3 scripts/data/build_vybor_from_xlsx.py
py -3 scripts/data/train_phase2_improved.py
```
