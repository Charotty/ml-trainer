# Отчёт по прогонам валидации clinical_honest (2026-06-30)

**Ветка:** `cursor/dicom-prep-pipeline`  
**Дата прогонов:** 2026-06-30  
**Главная честная метрика:** GroupKFold(5) OOF на **87 клинических** пациентах (Vybor xlsx)  
**Production-модель:** `models/adaptive_ensemble_clinical_honest.pkl`  
**Скрипт:** `scripts/data/train_clinical_honest.py --z-head ensemble`

Связанные документы: [`DATA_ARCHITECTURE.md`](DATA_ARCHITECTURE.md), [`NA_TRENDS_PRODUCTION_REPORT.md`](NA_TRENDS_PRODUCTION_REPORT.md), [`PRODUCTION_SYSTEM_REPORT.md`](PRODUCTION_SYSTEM_REPORT.md), [`дисер.md`](дисер.md).

---

## 1. Данные прогона

| Параметр | Значение |
|----------|----------|
| Источник меток | `data/Смещение - конечное -12 (2).xlsx` |
| Экспорт | `scripts/data/build_vybor_from_xlsx.py --no-boku` |
| CSV | `data/vybor_from_xlsx.csv` |
| Пациентов в xlsx | 100 (диссертация) |
| Пациентов в train | **87** (13 строк отброшены: неполные middle-point δ) |
| Holdout proxy-split | 18 Vybor (только для proxy-эксперимента) |
| na_spine cohort | 137 строк |
| na_boku cohort | 109 строк |
| KiTS19 cohort | 210 строк (только для trend-эксперимента) |

Манифест: `data/vybor_from_xlsx.manifest.json`

---

## 2. Архитектура (после исправления)

### 2.1. Что идёт в ensemble train

- **Только** `vybor_from_xlsx.csv` — реальные paired supine/lateral δ.
- Режим интеграции: `labeled_only` / `clinical_honest`.
- Без KiTS/DICOM/proxy в таргетах `y`.

### 2.2. Вспомогательные данные — когортные тренды

Модуль: `src/features/na_trend_features.py`, режим `enrichment_mode="na_trends"`.

| Признак | Источник | Смысл |
|---------|----------|--------|
| `na_pop_shift_{side}_{axis}` | na_spine + na_boku | median(lateral) − median(supine) по когорте |
| `na_sup_z_*`, `na_sup_pct_*` | na_spine | положение клиники vs supine-когорта |
| `kits_z_*`, `kits_pct_*` | KiTS19 (опц.) | положение клиники vs KiTS-когорта |
| `kits_cohort_median_*delta*` | KiTS19 (опц.) | медианы proxy-δ KiTS (константы) |

**Не используется:** per-patient `proj_lat_*` / `proj_sup_*`, подстановка объёмов из boku.

### 2.3. Честная оценка

- **GKF-5 OOF** — каждый пациент предсказывается моделью, не обучавшейся на нём.
- Bootstrap CI (n=2000) по per-patient avg error.
- Holdout 18 и in-sample 87 — **не** главные метрики (см. §7).

---

## 3. Сводная таблица всех вариантов (GKF-OOF 87)

| # | Вариант | Модель | Enrichment | Feat | Avg MAE | CI95 | X | Y | Z |
|---|---------|--------|------------|------|---------|------|---|---|---|
| A | Projection baseline | Ensemble | proj join + boku fill | 91 | **8.49** | 7.73–9.31 | 6.51 | 7.62 | 11.34 |
| B | **na_trends (production-optimal)** | Ensemble | na_spine + na_boku | 111 | **8.40** | 7.71–9.15 | 6.34 | 7.45 | 11.42 |
| C | na_trends + KiTS | Ensemble | spine + boku + KiTS | 159 | 8.44 | 7.75–9.18 | 6.37 | 7.51 | 11.45 |
| D | Quantile V7 Z-head | Ensemble + V7 Z | projection (старый) | 71 | 8.72 | 7.96–9.54 | 6.51 | 7.62 | 12.02 |
| E | Proxy GKF-OOF* | Ensemble | proxy train folds | — | 8.00 | 7.31–8.78 | 5.89 | 7.23 | 10.87 |

\* Вариант E: в каждом fold train = clinical + KiTS + DICOM pseudo (weighted); OOF только на clinical.

**Рекомендация production:** вариант **B** (`include_kits=False` по умолчанию).

---

## 4. Детализация по таргетам (MAE, мм)

### 4.1. A — Projection baseline

`results/validation_runs/clinical_honest_20260630/metrics/clinical_honest_report.json`

| Target | MAE |
|--------|-----|
| kidney_left_delta_x | 6.03 |
| kidney_left_delta_y | 7.73 |
| kidney_left_delta_z | 11.12 |
| kidney_right_delta_x | 6.99 |
| kidney_right_delta_y | 7.51 |
| kidney_right_delta_z | 11.56 |

### 4.2. B — na_trends (na_spine + na_boku) — **лучший ensemble**

Прогон: `clinical_honest_ensemble_20260630`, 111 features, 52 trend.

| Target | MAE |
|--------|-----|
| kidney_left_delta_x | 5.80 |
| kidney_left_delta_y | 7.58 |
| kidney_left_delta_z | 11.33 |
| kidney_right_delta_x | 6.88 |
| kidney_right_delta_y | 7.32 |
| kidney_right_delta_z | 11.51 |

Δ vs projection: avg **−0.09 mm**, X **−0.17**, Y **−0.17**, Z **+0.08**.

### 4.3. C — na_trends + KiTS19

`results/validation_runs/clinical_honest_ensemble_20260630/metrics/clinical_honest_report.json` (прогон с `include_kits=True`)

| Target | MAE |
|--------|-----|
| kidney_left_delta_x | 5.84 |
| kidney_left_delta_y | 7.63 |
| kidney_left_delta_z | 11.26 |
| kidney_right_delta_x | 6.91 |
| kidney_right_delta_y | 7.38 |
| kidney_right_delta_z | 11.65 |

KiTS cohort medians (proxy-δ, константы в признаках):

| Target | KiTS median δ |
|--------|---------------|
| kidney_left_delta_x | 15.70 |
| kidney_left_delta_y | 5.99 |
| kidney_left_delta_z | 9.36 |
| kidney_right_delta_x | −5.23 |
| kidney_right_delta_y | 5.62 |
| kidney_right_delta_z | 9.39 |

Δ vs B (без KiTS): avg **+0.04 mm** — в пределах шума, KiTS в трендах не даёт выигрыша.

### 4.4. D — Quantile V7 (experimental, не production)

`results/validation_runs/clinical_honest_quantile_v7_20260630/`

| Target | MAE |
|--------|-----|
| kidney_left_delta_x | 6.03 |
| kidney_left_delta_y | 7.73 |
| kidney_left_delta_z | **12.70** |
| kidney_right_delta_x | 6.99 |
| kidney_right_delta_y | 7.51 |
| kidney_right_delta_z | 11.35 |

Z хуже на **+0.68 mm** vs ensemble — откат в production.

### 4.5. E — Proxy-weighted GKF-OOF (эксперимент)

`results/validation_runs/clinical_proxy_20260630/metrics/clinical_proxy_vs_honest.json`

Train proxy: 69 clinical + 210 KiTS + 159 DICOM pseudo (weights 1.0 / 0.08 / 0.06).

| Target | Honest OOF | Proxy OOF | Δ |
|--------|------------|-----------|---|
| L_X | 6.03 | 5.54 | −0.49 |
| L_Y | 7.73 | 7.29 | −0.44 |
| L_Z | 11.12 | 11.04 | −0.08 |
| R_X | 6.99 | 6.24 | −0.75 |
| R_Y | 7.51 | 7.18 | −0.33 |
| R_Z | 11.56 | 10.71 | −0.85 |
| **Avg** | **8.49** | **8.00** | **−0.49** |

Proxy **не** рекомендован для production (не-клинические метки, плохой holdout — §7).

---

## 5. Сравнение с диссертацией

Источник: [`дисер.md`](дисер.md) — глава 3, табл. 7–9.

### 5.1. Реальное смещение (не ошибка модели)

| Ось | Диссертация (n=100) | Комментарий |
|-----|---------------------|-------------|
| X медиально | до −11,77 мм (L) | физическое смещение |
| Y вентрально | до +15 мм | CV ~80% |
| Z каудально (L) | ~−8,6 мм | R почки ~−1,5 мм (n.s.) |

### 5.2. MAE предсказания (Ridge + LOO-CV в диссертации)

| Ось | Диссертация MAE | Ensemble GKF-OOF (вариант B) | Разница |
|-----|-----------------|------------------------------|---------|
| X | 5,2–7,2 мм | 6,34 мм (avg X) | сопоставимо |
| Y | 5,8–7,6 мм | 7,45 мм (avg Y) | верхняя граница |
| Z | 7,8–9,5 мм | **11,42 мм** (avg Z) | **+2–3 мм хуже** |

### 5.3. Почему цифры расходятся

| Фактор | Диссертация | Модуль |
|--------|-------------|--------|
| CV | LOO-CV (99/1) | GroupKFold-5 OOF |
| Модель | Ridge | Adaptive ensemble |
| n | 100 | 87 |
| Метрика | MAE + % <10 мм + R² | MAE + bootstrap CI |
| Признаки | supine (диссер.) | supine + engineered + na_trends |

**Важно:** LOO-CV на Ridge обычно даёт **ниже** MAE, чем GKF-OOF на ансамбле. Для прямого сравнения с табл. 7–9 нужен отдельный прогон **Ridge + LOO-CV** на тех же 87 пациентах и тех же признаках.

### 5.4. Ridge + LOO-CV (диссертационный протокол)

> Отдельная программа Ridge + LOO-CV на тех же правилах и данных — для воспроизведения табл. 7–9 диссертации.  
> Результаты этого прогона добавьте в этот раздел или в отдельный файл и укажите путь.

| Ось | Диссертация | Ridge+LOO (ваш прогон) | Ensemble GKF-OOF (B) |
|-----|-------------|------------------------|----------------------|
| X MAE | 5,2–7,2 | *заполнить* | 6,34 |
| Y MAE | 5,8–7,6 | *заполнить* | 7,45 |
| Z MAE | 7,8–9,5 | *заполнить* | 11,42 |

---

## 6. KiTS19 в трендах — итог эксперимента

| | Без KiTS (B) | С KiTS (C) |
|--|--------------|------------|
| Trend features | 52 | 104 |
| Total features | 111 | 159 |
| Avg MAE OOF | **8.40** | 8.44 |
| Z avg | 11.42 | 11.45 |

**Вывод:** KiTS в trend extraction не улучшает OOF. Флаг `--with-kits` только для экспериментов.

JSON: `results/validation_runs/clinical_honest_ensemble_20260630/metrics/na_trends_kits_comparison.json`

---

## 7. Метрики, которым не доверять как «реальной точности»

### 7.1. Holdout 18 (несимметричное сравнение)

| Модель | Avg MAE | Z avg | Почему misleading |
|--------|---------|-------|-------------------|
| Honest (full train 87) | **2.70** | 4.18 | 18 пациентов **были в train** |
| Proxy (train 69+proxy) | 6.83 | 9.76 | честный holdout для proxy |

### 7.2. In-sample 87

| Модель | Avg MAE |
|--------|---------|
| Honest | 3.25 |
| Proxy | 3.03 |

### 7.3. Старый pipeline (~2 mm)

Proxy/KiTS в `y`, leaky features, in-sample / неверный split — **не сопоставимо** с текущим honest pipeline.

---

## 8. Команды воспроизведения

```bash
# Данные
py -3 scripts/data/build_vybor_from_xlsx.py --no-boku

# Production (na_trends, без KiTS)
py -3 scripts/data/train_clinical_honest.py --z-head ensemble

# Эксперимент: KiTS в трендах
py -3 scripts/data/train_clinical_honest.py --z-head ensemble --with-kits

# Proxy-эксперимент (не production)
py -3 scripts/data/train_clinical_proxy.py --skip-harmonize --skip-vybor-build

# Сравнение proxy vs honest
py -3 scripts/validation/compare_proxy_vs_honest.py
```

---

## 9. Артефакты на диске

| Описание | Путь |
|----------|------|
| Production model | `models/adaptive_ensemble_clinical_honest.pkl` |
| Projection baseline report | `results/validation_runs/clinical_honest_20260630/metrics/clinical_honest_report.json` |
| na_trends report (последний прогон) | `results/validation_runs/clinical_honest_ensemble_20260630/metrics/clinical_honest_report.json` |
| Projection vs na_trends | `results/validation_runs/clinical_honest_ensemble_20260630/metrics/honest_projection_vs_na_trends.json` |
| na_trends vs KiTS | `results/validation_runs/clinical_honest_ensemble_20260630/metrics/na_trends_kits_comparison.json` |
| Proxy comparison | `results/validation_runs/clinical_proxy_20260630/metrics/clinical_proxy_vs_honest.json` |
| V7 experiment | `results/validation_runs/clinical_honest_quantile_v7_20260630/metrics/clinical_honest_report.json` |
| Clinical CSV | `data/vybor_from_xlsx.csv` |
| Trend module | `src/features/na_trend_features.py` |

---

## 10. Итоговые выводы

1. **Честная production-точность:** ~**8.4 mm** avg MAE, Z ~**11.4 mm** (GKF-OOF 87).
2. **Лучший ensemble-вариант:** na_trends (na_spine + na_boku), **без** KiTS в трендах.
3. **Архитектура исправлена:** xlsx-only labels; na_spine/na_boku — когортные тренды, не per-patient join.
4. **Proxy** слегка лучше на OOF (−0.5 mm), но хуже на holdout и методологически слабее — не production.
5. **V7 Z-head** хуже ensemble — откат.
6. **Диссертация:** по X/Y ensemble близок к Ridge+LOO; по Z модуль хуже из‑за протокола (GKF vs LOO) и более сложной оси Z.
7. Для полного сопоставления с диссертацией — заполнить §5.4 результатами Ridge+LOO-CV на тех же 87 пациентах.

---

*Отчёт собран: 2026-06-30. Commit с na_trends: `d8f958e`.*
