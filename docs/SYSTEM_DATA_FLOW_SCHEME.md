# Схема: от исследовательской модели к программной системе

**Цель подраздела:** показать, как исследовательская постановка (МСКТ supine → предсказание смещения почек при переходе в lateral) реализована в текущем production-пайплайне репозитория `ml-trainer`.

**Production-модель:** `models/adaptive_ensemble_clinical_honest.pkl`  
**Тренер:** `models/phase1/adaptive_ensemble.py`  
**Обучение:** `scripts/data/train_clinical_honest.py`

---

## 1. Поток данных (inference — новый пациент)

На входе — **только МСКТ в положении лёжа (supine)**. Lateral-скан на этапе предсказания **не требуется**.

```mermaid
flowchart TB
    subgraph input["1. Вход"]
        DICOM["МСКТ DICOM\n(положение лёжа, supine)"]
    end

    subgraph extract["2. Извлечение анатомических признаков"]
        EXTRACT["scripts/inference/extract_from_dicom.py\nscripts/inference/enhanced_ct_extractor.py"]
        TS["TotalSegmentator:\nпочки, позвоночник, тело"]
        HARM["harmonize_extracted_datasets.py\n→ система координат Vybor"]
        SCHEMA["src/features/phase1_schema.py\nnormalize_dataframe()"]
        DICOM --> EXTRACT --> TS --> HARM --> SCHEMA
    end

    subgraph fe["3. Формирование признакового пространства"]
        BASE["BASE_FEATURES (23):\nrel-координаты почек, объёмы,\nрасстояния до spine, body COM"]
        ENG["Engineered (12):\nbody_ratio, volume_norm, asymmetry…"]
        CROSS["Cross-features (15):\nплотность, spine interaction…"]
        AXIS["Displacement-axis (leakage-safe):\nsupine lordosis / tilt / depth"]
        FILTER["filter_model_features()\nDROP: delta_span, lateral, proj_diff_*"]
        SCHEMA --> BASE --> ENG --> CROSS --> AXIS --> FILTER
    end

    subgraph trends["4. Когортные статистические признаки"]
        STORE["NaTrendStore\nsrc/features/na_trend_features.py"]
        SPINE["na_spine_full_aligned.csv\n137 supine CT"]
        BOKU["na_boku_full_aligned.csv\n109 lateral CT"]
        TREND["na_pop_shift_*\nna_sup_z_* / na_sup_pct_*\n(опц. kits_z_* при --with-kits)"]
        SPINE --> STORE
        BOKU --> STORE
        STORE --> TREND
        FILTER --> TREND
    end

    subgraph model["5. Адаптивная ансамблевая модель"]
        IMP["SimpleImputer (median)\nfit при обучении"]
        SCL["StandardScaler\nfit при обучении"]
        ENS["VotingRegressor × 6 таргетов\nRF + Lasso + Ridge + GBT\nвеса — GroupKFold по пациенту"]
        PKL["adaptive_ensemble_clinical_honest.pkl"]
        TREND --> IMP --> SCL --> ENS
        PKL -.->|загрузка весов| ENS
    end

    subgraph out["6. Выход"]
        PRED["Предсказание смещения (мм):\nΔX, ΔY, ΔZ — левая почка\nΔX, ΔY, ΔZ — правая почка"]
        API["Inference:\nsrc/features/pipeline.py\nscripts/validation/common.py predict_df()"]
        ENS --> PRED
        API --> PRED
    end
```

---

## 2. Поток обучения (отдельный контур меток)

Метки смещения **не** извлекаются из DICOM inference-пайплайна — они приходят из клинической таблицы с paired supine+lateral измерениями.

```mermaid
flowchart LR
    subgraph labels["Клинические метки y (единственный источник)"]
        XLSX["Excel:\nСмещение - конечное -12.xlsx\n~100 пациентов"]
        BUILD["build_vybor_from_xlsx.py\n--no-boku"]
        CSV["vybor_from_xlsx.csv\n87 пациентов с полными δ"]
        XLSX --> BUILD --> CSV
    end

    subgraph train["Обучение"]
        TR["train_clinical_honest.py\nenrichment_mode = na_trends"]
        GKF["GroupKFold(5) OOF\nоценка ~8.4 mm MAE"]
        CSV --> TR --> GKF
        TR --> PKL["clinical_honest.pkl"]
    end

    subgraph aux["Вспомогательные когорты (только тренды, не y)"]
        NASP["na_spine"]
        NABK["na_boku"]
        NASP --> TR
        NABK --> TR
    end
```

---

## 3. UML: компоненты системы

```mermaid
classDiagram
    class DicomExtract {
        +extract_from_dicom.py
        +enhanced_ct_extractor.py
        +TotalSegmentator masks
    }
    class Phase1Schema {
        +BASE_FEATURES
        +TARGET_NAMES
        +normalize_dataframe()
    }
    class NaTrendStore {
        +fit(spine, boku)
        +attach(df)
        +na_pop_shift_*
        +na_sup_z_*
    }
    class AdaptiveEnsembleTrainer {
        +prepare_training_data_split()
        +build_inference_matrix()
        +train_and_evaluate_adaptive_ensembles()
        +enrichment_mode: na_trends
    }
    class ClinicalHonestTrain {
        +train_clinical_honest.py
        +GKF OOF evaluation
    }
    class ModelArtifact {
        +clinical_honest.pkl
        +6 × VotingRegressor
        +imputer + scaler
        +na_trend_store
    }
    class InferenceAPI {
        +pipeline.py
        +predict_df()
        +kidney_displacement_api.py
    }

    DicomExtract --> Phase1Schema : supine geometry
    Phase1Schema --> AdaptiveEnsembleTrainer : base matrix
    NaTrendStore --> AdaptiveEnsembleTrainer : cohort trends
    ClinicalHonestTrain --> AdaptiveEnsembleTrainer : fit on vybor CSV
    AdaptiveEnsembleTrainer --> ModelArtifact : joblib dump
    ModelArtifact --> InferenceAPI : predict ΔXΔYΔZ
```

---

## 4. Соответствие этапов описанию подраздела

| Этап (текст подраздела) | Реализация в системе |
|-------------------------|----------------------|
| МСКТ (положение лёжа) | DICOM supine → `extract_from_dicom.py` |
| Извлечение анатомических признаков | TotalSegmentator + `phase1_schema` + `coordinate_harmonization` |
| Формирование признакового пространства | `AdaptiveEnsembleTrainer`: BASE + engineered + cross + axis (без утечек) |
| Когортные статистические признаки | `NaTrendStore` ← `na_spine` + `na_boku` (+ опц. KiTS) |
| Адаптивная ансамблевая модель | 6 × `VotingRegressor`, веса по GroupKFold |
| Предсказание ΔX, ΔY, ΔZ | 6 скалярных таргетов → 3D-вектор на почку |

---

## 5. Что сознательно **не** входит в production-поток

| Исключено | Причина |
|-----------|---------|
| KiTS19 / DICOM в таргетах `y` | proxy-метки, не клиническая истина |
| `proj_lat_*` join по ФИО | подмена per-patient lateral-координат |
| `proj_diff_*`, `delta_span`, `lateral` | утечка lateral-информации |
| Holdout-18 как главная метрика | optimistic при full-train |
| `adaptive_ensemble.pkl` (legacy API) | старая модель без honest-пайплайна |

---

## 6. Команды

```powershell
# Обучение production
py -3 scripts/data/build_vybor_from_xlsx.py --no-boku
py -3 scripts/data/train_clinical_honest.py --z-head ensemble

# Inference (тот же FE, что при обучении)
# predict_df(bundle, supine_features_df)  →  6 колонок delta_*
```

См. также: [`DATA_ARCHITECTURE.md`](DATA_ARCHITECTURE.md), [`CLINICAL_VALIDATION_RUN_REPORT_20260630.md`](CLINICAL_VALIDATION_RUN_REPORT_20260630.md).
