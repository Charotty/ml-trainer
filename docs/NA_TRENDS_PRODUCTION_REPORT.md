# Отчёт: production-модель и когортные тренды (na_trends)

**Дата:** 2026-06-30  
**Ветка:** `cursor/dicom-prep-pipeline`  
**Production-модель:** `models/adaptive_ensemble_clinical_honest.pkl`  
**Скрипт обучения:** `scripts/data/train_clinical_honest.py --z-head ensemble`

---

## 1. Резюме

Production-модель предсказывает 3D-смещение почек (ΔX, ΔY, ΔZ, мм) при переходе **supine → lateral** по **только supine-признакам**.

| Показатель | Значение |
|------------|----------|
| Клинических пациентов | 87 |
| Честная метрика | GroupKFold(5) OOF по пациенту |
| **Avg MAE (production, na_trends)** | **8.40 mm** [7.71 – 9.15] *(best: spine+boku only)* |
| **Z avg MAE** | **11.42 mm** |
| Признаков | 111 (52 trend) / 159 (104 trend + KiTS) |

---

## 2. Архитектура данных (исправленная)

### 2.1. Главная таблица — единственный источник меток

```
Excel (Смещение - конечное -12.xlsx)
        ↓  build_vybor_from_xlsx.py --no-boku
data/vybor_from_xlsx.csv   ← 87 пациентов, paired supine + lateral, реальные δ
        ↓  train_clinical_honest.py
models/adaptive_ensemble_clinical_honest.pkl
```

**Только этип xlsx → CSV → ensemble** задаёт таргеты для regression.

### 2.2. Вспомогательные CT — когортные тренды (не метки)

| Источник | Строк | Роль |
|----------|-------|------|
| `na_spine_full.csv` | 137 | supine-когорта: распределения анатомии |
| `na_boku_full.bak.csv` | 109 | lateral-когорта: распределения анатомии |
| KiTS19 (опционально) | 210 | внешняя когорта: анатомия + **медианы proxy-δ** |

**Не используется в honest production:**
- KiTS/DICOM как строки train с proxy-таргетами
- `proj_lat_*` / `proj_sup_*` join по ФИО
- подстановка объёмов из boku в xlsx

**Используется (`src/features/na_trend_features.py`):**

| Признак | Смысл |
|---------|--------|
| `na_pop_shift_{side}_{axis}` | median(бок) − median(спина) по когорте |
| `na_sup_z_*`, `na_sup_pct_*` | z-score / percentile клиники vs na_spine |
| `kits_z_*`, `kits_pct_*` | z-score / percentile клиники vs KiTS19 |
| `kits_cohort_median_*delta*` | медианы proxy-δ KiTS по когорте (константы) |

---

## 3. Сравнение протоколов оценки

| Протокол | Honest avg | Комментарий |
|----------|------------|-------------|
| **GKF-OOF 87** | **~8.4 mm** | ✅ главная метрика |
| Holdout 18 (full-train) | ~2.7 mm | ❌ optimistic — модель видела пациентов |
| In-sample 87 | ~3.3 mm | ❌ переобучение |
| Старый pipeline + proxy/KiTS в y | ~2 mm | ❌ не сопоставимо (утечки + proxy labels) |

---

## 4. Сравнение вариантов enrichment (GKF-OOF 87)

### 4.1. Projection (старый) vs na_trends (na_spine + na_boku)

| Метрика | Projection (91 feat) | na_trends (111 feat) | Δ |
|---------|----------------------|----------------------|---|
| **Avg MAE** | 8.49 mm | **8.40 mm** | **−0.09** |
| X | 6.51 | **6.34** | −0.17 |
| Y | 7.62 | **7.45** | −0.17 |
| Z | **11.34** | 11.42 | +0.08 |
| CI95 avg | 7.73 – 9.31 | 7.71 – 9.15 | пересекаются |

**По таргетам (na_trends):**

| Target | Projection | na_trends |
|--------|------------|-----------|
| L_X | 6.03 | 5.80 |
| L_Y | 7.73 | 7.58 |
| L_Z | 11.12 | 11.33 |
| R_X | 6.99 | 6.88 |
| R_Y | 7.51 | 7.32 |
| R_Z | 11.56 | 11.51 |

### 4.2. na_trends + KiTS19 (эксперимент)

| Метрика | na_trends (без KiTS) | na_trends + KiTS | Δ (KiTS − без) |
|---------|----------------------|------------------|----------------|
| **Avg MAE** | **8.40 mm** | 8.44 mm | +0.04 |
| X | 6.34 | 6.37 | +0.03 |
| Y | 7.45 | 7.51 | +0.06 |
| Z | 11.42 | 11.45 | +0.03 |
| Trend features | 52 | 104 | +52 |
| Total features | 111 | 159 | +48 |

**По таргетам (na_trends + KiTS):**

| Target | без KiTS | + KiTS |
|--------|----------|--------|
| L_X | 5.80 | 5.84 |
| L_Y | 7.58 | 7.63 |
| L_Z | 11.33 | 11.26 |
| R_X | 6.88 | 6.91 |
| R_Y | 7.32 | 7.38 |
| R_Z | 11.51 | 11.65 |

KiTS добавляет:
- 46 признаков `kits_z_*` / `kits_pct_*` (анатомия vs KiTS-когорта, 210 строк)
- 6 констант `kits_cohort_median_*delta*` (медианы proxy-δ KiTS)

**Вывод:** KiTS в трендах **не улучшает** OOF (~+0.04 mm avg — шум). Лучший вариант — **na_spine + na_boku без KiTS**.

---

## 5. Proxy-эксперiment (не production)

Weighted train: Vybor + KiTS proxy + DICOM pseudo (`train_clinical_proxy.py`).

| Протокол | Honest | Proxy |
|----------|--------|-------|
| GKF-OOF 87 | 8.49 mm | 8.00 mm |
| Holdout 18 | 2.70 mm | 6.83 mm |

Proxy чуть лучше на OOF, но хуже на holdout и использует не-клинические метки → **не production**.

---

## 6. Pipeline обучения

```bash
py -3 scripts/data/build_vybor_from_xlsx.py --no-boku
py -3 scripts/data/train_clinical_honest.py --z-head ensemble
# эксперимент с KiTS в трендах:
py -3 scripts/data/train_clinical_honest.py --z-head ensemble --with-kits
```

**Внутри train_clinical_honest:**
1. `NaTrendStore.fit()` — статистики na_spine + na_boku (+ KiTS)
2. `enrichment_mode="na_trends"` — без projection join
3. Full fit на 87 clinical
4. GKF-5 OOF для отчёта

---

## 7. Inference

Модель pickle содержит `na_trend_store` — те же когортные константы при predict.

```python
# bundle fields: enrichment_mode, na_trend_store, feature_names, models, imputer, scaler
```

FastAPI (`src/api/kidney_displacement_api.py`) по умолчанию грузит старый `adaptive_ensemble.pkl` — для production явно укажите `clinical_honest.pkl`.

---

## 8. Выводы

1. **~8.4 mm GKF-OOF** — реалистичная точность на 87 клинических пациентах.
2. **na_trends** архитектурно корректнее projection join; метрики не хуже (слегка лучше X/Y).
3. **Z (~11.4 mm)** остаётся узким местом.
4. **KiTS в трендах** — эксперимент: +52 признака, OOF **не лучше** (+0.04 mm). Оставить KiTS опциональным; production-optimal — spine+boku only.

---

## 9. Артефакты

| Артефакт | Путь |
|----------|------|
| Production model | `models/adaptive_ensemble_clinical_honest.pkl` |
| Отчёт na_trends | `results/validation_runs/clinical_honest_ensemble_20260630/` |
| Сравнение projection vs na_trends | `results/validation_runs/clinical_honest_ensemble_20260630/metrics/honest_projection_vs_na_trends.json` |
| Архитектура данных | `docs/DATA_ARCHITECTURE.md` |
| KiTS+trends comparison | `results/validation_runs/clinical_honest_ensemble_20260630/metrics/na_trends_kits_comparison.json` |
