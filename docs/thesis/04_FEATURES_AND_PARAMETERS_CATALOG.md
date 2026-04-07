# 04. Каталог признаков и параметров (Features & Parameters Catalog)

## 0. Зачем нужен этот документ
В диссертации обычно недостаточно сказать «мы использовали 51 признак». Нужны:
- формулы;
- единицы измерения;
- клиническая интерпретация;
- роль признака в модели;
- риски (NaN, деление на ноль, утечки, несогласованность train/inference).

Этот документ фиксирует **актуальный production-контракт** (Phase 1 интегрированного ансамбля), который реально используется в `src/api/kidney_displacement_api.py`.

Источник «истины» по формулам:
- `models/phase1/adaptive_ensemble.py`:
  - `required_features`
  - `_create_engineered_features`
  - `_create_cross_features`

## 1. Базовые признаки (required_features)

### 1.1. Группа: координаты центров почек (relative)

#### `kidney_left_center_x_rel`
- **Тип**: float
- **Единицы**: мм
- **Смысл**: относительная X-координата центра левой почки в принятой системе координат пациента.
- **Почему важно**: по положению в X часто определяется латеральность и “рычаг” смещения при повороте пациента.

#### `kidney_left_center_y_rel`
- **Тип**: float
- **Единицы**: мм
- **Смысл**: относительная Y-координата центра левой почки.

#### `kidney_left_center_z_rel`
- **Тип**: float
- **Единицы**: мм
- **Смысл**: относительная Z-координата центра левой почки.

#### `kidney_right_center_x_rel`, `kidney_right_center_y_rel`, `kidney_right_center_z_rel`
Аналогично для правой почки.

### 1.2. Группа: размеры почек

#### `kidney_left_length_mm`
- **Тип**: float
- **Единицы**: мм
- **Смысл**: длина левой почки (как анатомическая характеристика, извлечённая из DICOM или табличной записи).

#### `kidney_left_volume_cm3`
- **Тип**: float
- **Единицы**: см³
- **Смысл**: объём левой почки.

#### `kidney_right_length_mm`, `kidney_right_volume_cm3`
Аналогично для правой почки.

### 1.3. Группа: геометрия тела

#### `body_width_mm`
- **Тип**: float
- **Единицы**: мм
- **Смысл**: ширина тела пациента (в поперечном сечении).
- **Роль**: используется как нормализатор (делитель) для ряда engineered/cross признаков.

#### `body_depth_mm`
- **Тип**: float
- **Единицы**: мм
- **Смысл**: “глубина” тела пациента.

#### `body_area_mm2`
- **Тип**: float
- **Единицы**: мм²
- **Смысл**: площадь поперечного сечения тела.
- **Роль**: используется для `volume_to_area_ratio_*`.

### 1.4. Группа: расстояния до опорных структур

#### `kidney_left_to_spine_distance`, `kidney_right_to_spine_distance`
- **Тип**: float
- **Единицы**: мм
- **Смысл**: расстояние от соответствующей почки до позвоночника.
- **Интерпретация**: прокси “степени фиксации”/положения почки относительно оси тела.

#### `kidney_left_to_body_center_distance`, `kidney_right_to_body_center_distance`
- **Тип**: float
- **Единицы**: мм
- **Смысл**: расстояние от почки до центра масс тела.

### 1.5. Группа: координаты центра позвоночника и центра масс тела

#### `spine_center_x`, `spine_center_y`, `spine_center_z`
- **Тип**: float
- **Единицы**: мм
- **Смысл**: координаты центра позвоночника.
- **Примечание**: в API эти поля имеют default=0.0, что означает, что при отсутствии данных используется “нулевая система” (упрощение).

#### `body_com_x`, `body_com_y`, `body_com_z`
- **Тип**: float
- **Единицы**: мм
- **Смысл**: координаты центра масс тела.

## 2. Инженерные признаки (engineered_features, 13)
Создаются функцией `_create_engineered_features(df)`.

### 2.1. `body_ratio`
- **Формула**: `body_width_mm / body_depth_mm`
- **Смысл**: отношение ширины к глубине (форма тела).
- **Риск**: деление на ноль, если `body_depth_mm=0` (на практике должно быть предотвращено валидацией).

### 2.2. `kidney_distance_lr`
- **Формула**: `abs(kidney_left_center_x_rel - kidney_right_center_x_rel)`
- **Смысл**: расстояние между почками по оси X (упрощённая латеральная сепарация).

### 2.3. Нормализованные размеры почек

#### `kidney_left_volume_norm`
- **Формула**: `kidney_left_volume_cm3 / body_width_mm`
- **Смысл**: объём левой почки, нормализованный на размер тела.

#### `kidney_right_volume_norm`
- **Формула**: `kidney_right_volume_cm3 / body_width_mm`

#### `kidney_left_length_norm`
- **Формула**: `kidney_left_length_mm / body_width_mm`

#### `kidney_right_length_norm`
- **Формула**: `kidney_right_length_mm / body_width_mm`

### 2.4. Асимметрии (лево-право)

#### `volume_asymmetry`
- **Формула**: `kidney_left_volume_cm3 - kidney_right_volume_cm3`

#### `length_asymmetry`
- **Формула**: `kidney_left_length_mm - kidney_right_length_mm`

#### `spine_distance_asymmetry`
- **Формула**: `kidney_left_to_spine_distance - kidney_right_to_spine_distance`

#### `body_center_asymmetry`
- **Формула**: `kidney_left_to_body_center_distance - kidney_right_to_body_center_distance`

Смысл асимметрий:
- они могут отражать анатомические особенности, которые влияют на смещение при смене положения.

### 2.5. Нормализованные расстояния

#### `kidney_left_to_spine_ratio`
- **Формула**: `kidney_left_to_spine_distance / body_width_mm`

#### `kidney_right_to_spine_ratio`
- **Формула**: `kidney_right_to_spine_distance / body_width_mm`

### 2.6. `patient_position_encoded`
- **Формула в текущей реализации**: если столбца нет, ставится `1` (supine)
- **Смысл**: категориальный индикатор положения пациента.
- **Важное ограничение**: если модель обучалась только на одном положении, этот признак может быть константой.

## 3. Cross-features (15)
Создаются функцией `_create_cross_features(df)`.

### 3.1. `body_volume_estimated`
- **Формула (как в коде)**:
  - `avg_kidney_height = (kidney_left_length_mm + kidney_right_length_mm)/2`
  - `body_volume_estimated = body_width_mm * body_depth_mm * avg_kidney_height / 1000`
- **Смысл**: грубая оценка объёма тела (в см³) через “толщину” порядка длины почки.

### 3.2. `kidney_left_density_ratio`, `kidney_right_density_ratio`
- **Формула**:
  - `kidney_left_volume_cm3 / kidney_left_length_mm`
  - `kidney_right_volume_cm3 / kidney_right_length_mm`
- **Смысл**: прокси «плотности/массы на длину».

### 3.3. `spine_to_body_ratio_x`, `spine_to_body_ratio_y`
- **Формула**:
  - `spine_center_x / body_width_mm`
  - `spine_center_y / body_depth_mm`
- **Смысл**: положение позвоночника в нормализованных координатах.

### 3.4. `body_com_to_spine_distance`
- **Формула**: `sqrt((body_com_x - spine_center_x)^2 + (body_com_y - spine_center_y)^2)`
- **Смысл**: расстояние между центром масс и позвоночником (в XY).

### 3.5. Взаимодействия (interaction terms)

#### `kidney_left_spine_interaction`
- **Формула**: `kidney_left_to_spine_distance * kidney_left_volume_cm3`

#### `kidney_right_spine_interaction`
- **Формула**: `kidney_right_to_spine_distance * kidney_right_volume_cm3`

### 3.6. Индексы размера/положения

#### `body_size_index`
- **Формула**: `sqrt(body_width_mm^2 + body_depth_mm^2)`

#### `kidney_position_index_left`
- **Формула**: `sqrt(x^2 + y^2 + z^2)` по `kidney_left_center_*_rel`

#### `kidney_position_index_right`
Аналогично.

### 3.7. Нормализация объёма на площадь

#### `volume_to_area_ratio_left`
- **Формула**: `kidney_left_volume_cm3 / (body_area_mm2 / 100)`
- **Комментарий по единицам**: `body_area_mm2/100` переводит мм² в см² (приближенно, т.к. 1 см² = 100 мм²).

#### `volume_to_area_ratio_right`
Аналогично.

### 3.8. `relative_volume_sum`
- **Формула**: `(kidney_left_volume_cm3 + kidney_right_volume_cm3) / body_width_mm`
- **Смысл**: суммарный объём почек относительно размера тела.

### 3.9. `kidney_separation_angle`
- **Формула (упрощённо, XY)**:
  - строятся вектора `left_vector=(x_left, y_left)`, `right_vector=(x_right, y_right)`
  - `cos = dot(left,right)/(||left||*||right||)`
  - `angle = arccos(clip(cos,-1,1)) * 180/pi`
- **Смысл**: “угол” расположения почек относительно начала координат.

## 4. Целевые переменные (targets)

### 4.1. Список
- `kidney_left_delta_x`, `kidney_left_delta_y`, `kidney_left_delta_z`
- `kidney_right_delta_x`, `kidney_right_delta_y`, `kidney_right_delta_z`

### 4.2. Как интерпретировать знак
Для диссертации рекомендуется добавить соглашение:
- положительный `delta_x` — смещение в сторону увеличения X (например, вправо)
- положительный `delta_y` — вперёд/назад (в зависимости от оси)
- положительный `delta_z` — вверх/вниз (в зависимости от оси)

Важно: знак зависит от конкретной системы координат. В проекте есть отдельные модули про систему координат (`src/coordinate_system.py` и др.), и в диссертации нужно сделать единый раздел «Coordinate System Definition».

## 5. Гиперпараметры моделей (Phase 1)

### 5.1. RandomForestRegressor
В `load_base_models()` задано:
- `n_estimators=500`
- `max_depth=20`
- `min_samples_split=10`
- `min_samples_leaf=4`
- `max_features='sqrt'`
- `random_state=42`
- `n_jobs=-1`

**Интерпретация основных параметров**:
- `n_estimators`: число деревьев. Больше — стабильнее, но дольше.
- `max_depth`: глубина дерева. Больше — риск переобучения.
- `min_samples_split`: минимальное число объектов для разбиения узла.
- `min_samples_leaf`: минимальное число объектов в листе.
- `max_features='sqrt'`: сколько признаков пробовать на сплите.
- `random_state`: воспроизводимость.

### 5.2. Lasso
- `alpha=0.1` — сила L1 регуляризации.
- `max_iter=5000` — итерации оптимизатора.

### 5.3. Ridge
- `alpha=1.0` — сила L2 регуляризации.
- `solver='auto'`

### 5.4. GradientBoostingRegressor
- `n_estimators=500`
- `learning_rate=0.05`
- `max_depth=5`
- `subsample=0.8`

## 6. Параметры оптимизации весов ансамбля
- оптимизация `L-BFGS-B`
- `bounds=(0,1)` для каждого веса
- `maxiter=100`

Для диссертации:
- привести постановку оптимизационной задачи математически.

## 7. Параметры API
См. `05_API_AND_INFERENCE_PIPELINE.md`, но здесь важно подчеркнуть:
- **контракт входа** (23 базовых признака)
- **полный список признаков для модели** задаётся `feature_names` в артефакте.

## 8. Проверки и валидация (что добавить, чтобы было «диссертационно»)
Рекомендуемые проверки (часть уже есть в проекте, часть можно описать как будущую работу):
- диапазоны значений (например, `body_width_mm > 0`)
- отсутствие NaN в базовых полях
- контроль единиц (мм vs см)
- контроль физической реализуемости (дельты в разумном диапазоне)

## 9. Как этот каталог использовать в тексте диссертации
- В основной главе: описать группы признаков и логику их построения.
- В приложениях: дать полный справочник признаков и параметров (этот документ).
- В разделе «воспроизводимость»: указать, что список признаков хранится в артефакте `adaptive_ensemble.pkl`.
