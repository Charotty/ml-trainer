# CT Workbench — анализ test 2 и рекомендации

Документ фиксирует результаты прогона **test 2** в CT Workbench, сопоставление с обучающим корпусом Vybor и практические шаги по исправлению масштаба признаков и предсказаний.

## Тестовые кейсы

| Кейс | patient_label | Пациент | case_id / источник |
|------|---------------|---------|-------------------|
| **test 2** | test 2 | Abramyan Khristofor Karenovich | 738d1bff-9de9-4e7f-9077-2226b5044ae4 (UI / Cases API) |
| **test 1 (сравнение)** | — | Agababova Irina Shagenovna | case_0003_514f7c035ef7, batch na_spine (Агабабова.zip) |

Для test 2: **719** срезов DICOM, TotalSegmentator успешен, объёмы почек в норме.

---

## Метрики инференса (test 2)

Источник: data/cases/738d1bff-9de9-4e7f-9077-2226b5044ae4/artifacts/prediction.json.

| Признак | Значение |
|---------|----------|
| kidney_left_delta_x | -82386492229130.44 mm |
| kidney_left_delta_y | 716731577439256.4 mm |
| kidney_left_delta_z | -18.49834903612908 mm |
| kidney_right_delta_x | -1854761878484411.2 mm |
| kidney_right_delta_y | 146800810489151.66 mm |
| kidney_right_delta_z | -8.019028632721085 mm |
| coverage_pct | 100% |
| model_id | adaptive_ensemble_clinical_honest.pkl |
| feature_count | 111 |
| enrichment_mode | na_trends |

Компоненты delta_z выглядят клинически правдоподобно (порядка −8…−18 mm). Компоненты delta_x / delta_y — явная экстраполяция вне обучающего распределения из‑за масштаба входных признаков.

---

## Ключевые признаки инференса (test 2)

Источник: artifacts/features.json → all_features.

### Относительные координаты центров почек (мм)

| Признак | Левая | Правая |
|---------|-------|--------|
| kidney_*_center_x_rel | 52.67 | 198.27 |
| kidney_*_center_y_rel | 162.16 | 160.94 |
| kidney_*_center_z_rel | 76.75 | 47.77 |

### Расстояния до позвоночника и ориентиры

| Признак | Значение |
|---------|----------|
| kidney_left_to_spine_distance | 187.0 mm |
| kidney_right_to_spine_distance | 259.8 mm |
| spine_center_z | 1585.7 mm |

### Объёмы и сегментация

| Признак | Значение |
|---------|----------|
| kidney_left_volume_cm3 | 192.8 |
| kidney_right_volume_cm3 | 189.0 |
| series_slices | 719 |
| TotalSegmentator | ok |
| coverage_pct | 100% |

### Каскадные производные (фрагмент)

- kidney_left_spine_interaction: ~36042.7
- kidney_right_spine_interaction: ~49094.5
- na_sup_pct_* / na_sup_z_* для координат и расстояний — также на порядки выше типичных значений Vybor после обогащения.

---

## Эталон обучения (Vybor, data/vybor_from_xlsx.csv, n=87)

Типичные диапазоны признаков, на которых обучена клиническая модель:

| Признак | min | median | max |
|---------|-----|--------|-----|
| kidney_left_center_x_rel | −11.6 | 1.65 | 11.9 |
| kidney_left_center_y_rel | −13.05 | −0.40 | 9.1 |
| kidney_left_center_z_rel | −33.29 | 3.45 | 24.7 |
| kidney_left_to_spine_distance | 1.80 | 8.84 | 34.19 |
| kidney_right_to_spine_distance | 1.80 | 8.84 | 34.19 |
| spine_center_z | −115.4 | −53.35 | −9.1 |

test 2 по rel-координатам, расстояниям до позвоночника и spine_center_z отклоняется на 1–3 порядка от этих диапазонов. test 1 (Agababova) в batch-пайплайне na_spine даёт признаки согласованного масштаба с обучением (в отличие от UI-пути test 2).

---

## Корневые причины

1. **Несовпадение системы координат:** модель обучена на Vybor Excel, где ориентир (spine) согласован с midpoint почек; инференс в Workbench использует сырые LPS-координаты DICOM без той же re-anchoring.
2. **Разные якоря сегментации:** позвоночник из enhanced_ct_extractor, почки из TotalSegmentator — разные опорные точки для relative-признаков.
3. **Нет гармонизации на UI-инференсе:** harmonize_extracted_datasets / harmonize_dataframe применяются в batch, но не перед build_inference_matrix в features_service.
4. **Каскад по engineered features:** interaction-поля, na_sup_pct_*, асимметрии усиливают ошибку масштаба.
5. **Поведение модели по осям:** регрессоры X/Y экстраполируют OOD; Z стабильнее (~−8…−18 mm).
6. **coverage 100% обманчив:** после фикса enrichment все 111 признаков заполнены, но численный масштаб остаётся неверным.

---

## Рекомендации

1. Вызывать harmonize_dataframe (опора на reference_stats.json) в features_service **до** build_inference_matrix.
2. Альтернатива/дополнение: re-anchor spine как midpoint почек, как в Excel Vybor.
3. После merge почек TotalSegmentator в extract_from_dicom вызывать merge_spine_relative (или эквивалентную нормализацию).
4. Повторить прогон test 2 и сравнить features.json / prediction.json с диапазонами Vybor.
5. В UI показывать предупреждение, если ключевые rel-признаки вне перцентилей обучения (даже при coverage 100%).

---

## Работа пайплайна в текущей сессии

- Поддержка извлечения из ZIP в подготовке DICOM (extraction_runner).
- Обучение adaptive_ensemble_clinical_honest.pkl на Vybor xlsx + доп. корпуса na_spine / na_boku.
- Enrichment 13 клинических признаков (na_trends) для полного покрытия матрицы модели.
- Автовыбор device (CPU/GPU) в worker для TotalSegmentator.
- Обновления train_clinical_honest.py и тестов enrichment.

---

## Артефакты

- Модель: models/adaptive_ensemble_clinical_honest.pkl (~24.6 MB, 25 788 369 байт).
- Кейс test 2: data/cases/738d1bff-9de9-4e7f-9077-2226b5044ae4 — **не коммитить** (локальные DICOM/артефакты).

*Дата документа: 2026-07-14.*
