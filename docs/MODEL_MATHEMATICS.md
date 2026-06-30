# Математическая постановка и методы production-модели

Документ описывает **только production-пайплайн** предсказания смещения почек:

- обучение: `models/phase1/adaptive_ensemble.py`
- контракт признаков: `src/features/phase1_schema.py`, `config/phase1_feature_schema.yaml`
- инференс: `src/features/pipeline.py`, `src/api/kidney_displacement_api.py`
- артефакт: `models/adaptive_ensemble.pkl`

---

## 1. Задача

Для каждого клинического случая \(i = 1,\ldots,N\) по **supine**-геометрии и анатомии тела предсказать **смещение центра почки** при переходе в **lateral** положение.

Вектор смещения (мм) в системе координат пациента (LPS):

\[
\boldsymbol{\delta}^{(L)}_i =
\begin{pmatrix}
\delta^{(L)}_{x,i} \\ \delta^{(L)}_{y,i} \\ \delta^{(L)}_{z,i}
\end{pmatrix},
\qquad
\boldsymbol{\delta}^{(R)}_i =
\begin{pmatrix}
\delta^{(R)}_{x,i} \\ \delta^{(R)}_{y,i} \\ \delta^{(R)}_{z,i}
\end{pmatrix}
\]

**Таргеты модели** (6 скалярных регрессий):

| Индекс | Переменная | Смысл |
|--------|------------|--------|
| 1 | `kidney_left_delta_x` | левая почка, ось X (L→R) |
| 2 | `kidney_left_delta_y` | левая почка, ось Y (P→A) |
| 3 | `kidney_left_delta_z` | левая почка, ось Z (I→S) |
| 4 | `kidney_right_delta_x` | правая почка, X |
| 5 | `kidney_right_delta_y` | правая почка, Y |
| 6 | `kidney_right_delta_z` | правая почка, Z |

Клиническая интерпретация одного компонента:

\[
\delta_{a,i} = p^{\mathrm{lateral}}_{a,i} - p^{\mathrm{supine}}_{a,i},
\quad a \in \{x,y,z\},\; \text{side} \in \{L,R\}
\]

где \(p\) — координата **центра почки** (обычно средняя точка по кранио-каудальной оси) в мм.

**Важно:** из одной CT-серии (только supine) таргеты **не извлекаются** — они приходят из клинической таблицы (Vybor / Excel). DICOM-экстрактор даёт только входные признаки \(\mathbf{x}_i\).

---

## 2. Обозначения признаков

### 2.1 Базовые признаки \(\mathbf{b}_i \in \mathbb{R}^{23}\)

Канонический вектор до инженерии (`BASE_FEATURES`):

- относительные координаты центров почек к позвоночнику:
  \[
  r^{L}_{a} = p^{L,\mathrm{supine}}_{a} - s_a,\quad
  r^{R}_{a} = p^{R,\mathrm{supine}}_{a} - s_a
  \]
  где \(s_a\) — компонента центра позвоночника (`spine_center_*`).
- объёмы и длины почек (`kidney_*_volume_cm3`, `kidney_*_length_mm`).
- геометрия тела (`body_width_mm`, `body_depth_mm`, `body_area_mm2`).
- расстояния почка–позвоночник, почка–центр тела.
- абсолютные `spine_center_*`, `body_com_*`.

Нормализация имён и алиасов — `normalize_dataframe()` в `src/features/phase1_schema.py`.

### 2.2 Инженерные признаки \(\mathbf{e}_i \in \mathbb{R}^{13}\)

Детерминированные преобразования \(\phi_{\mathrm{eng}}(\mathbf{b}_i)\):

| Признак | Формула |
|---------|---------|
| `body_ratio` | \(W_i / D_i\) |
| `kidney_distance_lr` | \(\|r^L_x - r^R_x\|\) |
| `kidney_*_volume_norm` | \(V_{\mathrm{kidney}} / W\) |
| `kidney_*_length_norm` | \(L_{\mathrm{kidney}} / W\) |
| `volume_asymmetry` | \(V^L - V^R\) |
| `length_asymmetry` | \(L^L - L^R\) |
| `spine_distance_asymmetry` | \(d^L_{\mathrm{spine}} - d^R_{\mathrm{spine}}\) |
| `body_center_asymmetry` | \(d^L_{\mathrm{body}} - d^R_{\mathrm{body}}\) |
| `kidney_*_to_spine_ratio` | \(d_{\mathrm{spine}} / W\) |
| `patient_position_encoded` | дискретный код DICOM-позы (HFS→1, FFS→2, …) |

### 2.3 Кросс-признаки \(\mathbf{c}_i \in \mathbb{R}^{15}\)

\(\phi_{\mathrm{cross}}(\mathbf{b}_i, \mathbf{e}_i)\):

| Признак | Формула |
|---------|---------|
| `body_volume_estimated` | \(W \cdot D \cdot \bar{L}_{\mathrm{kidney}} / 1000\) |
| `kidney_*_density_ratio` | \(V_{\mathrm{kidney}} / L_{\mathrm{kidney}}\) |
| `spine_to_body_ratio_x` | \(s_x / W\) |
| `spine_to_body_ratio_y` | \(s_y / D\) |
| `body_com_to_spine_distance` | \(\sqrt{(c_x-s_x)^2 + (c_y-s_y)^2}\) |
| `kidney_*_spine_interaction` | \(d_{\mathrm{spine}} \cdot V_{\mathrm{kidney}}\) |
| `body_size_index` | \(\sqrt{W^2 + D^2}\) |
| `kidney_position_index_*` | \(\| \mathbf{r}^{\mathrm{side}} \|_2\) |
| `volume_to_area_ratio_*` | \(V / (A/100)\) |
| `relative_volume_sum` | \((V^L + V^R) / W\) |
| `kidney_separation_angle` | \(\arccos\!\left(\frac{\mathbf{u}^L \cdot \mathbf{u}^R}{\|\mathbf{u}^L\|\|\mathbf{u}^R\|}\right)\cdot\frac{180}{\pi}\) |

Полный вектор признаков до препроцессинга:

\[
\mathbf{x}_i = \big[\mathbf{b}_i \;\|\; \mathbf{e}_i \;\|\; \mathbf{c}_i\big] \in \mathbb{R}^{d},
\quad d \approx 51
\]

Порядок колонок фиксируется при обучении в `feature_names` и воспроизводится на инференсе.

---

## 3. Препроцессинг (train = inference)

### 3.1 Импутация пропусков

`SimpleImputer(strategy='median')` — по каждому признаку \(j\):

\[
\tilde{x}_{ij} =
\begin{cases}
x_{ij}, & x_{ij} \text{ не NaN} \\
\mathrm{median}_j(\mathcal{D}_{\mathrm{train}}), & \text{иначе}
\end{cases}
\]

**Статистика медианы** считается **только на train** и сохраняется в pickle.

### 3.2 Стандартизация

`StandardScaler` — по train:

\[
\mu_j = \frac{1}{N_{\mathrm{train}}}\sum_{i \in \mathrm{train}} \tilde{x}_{ij},
\qquad
\sigma_j = \sqrt{\frac{1}{N_{\mathrm{train}}}\sum_{i}(\tilde{x}_{ij}-\mu_j)^2}
\]

\[
z_{ij} = \frac{\tilde{x}_{ij} - \mu_j}{\sigma_j}
\]

На validation и inference применяется тот же \((\mu_j, \sigma_j)\).

### 3.3 Порядок на инференсе

```text
normalize_record → build_inference_matrix → imputer.transform → scaler.transform → predict
```

Реализация: `predict_targets()` в `src/features/pipeline.py`.

---

## 4. Модель: шесть независимых регрессоров

**Нет** одной многомерной модели на все оси. Для каждого таргета \(t \in \mathcal{T}\) (6 штук) обучается **свой** регрессор \(f_t\).

\[
\hat{y}_{i,t} = f_t(\mathbf{z}_i)
\]

где \(\mathbf{z}_i\) — стандартизованный вектор признаков пациента \(i\).

---

## 5. Базовые регрессоры (общий пул)

Для **каждого** таргета \(t\) используется один и тот же набор из четырёх базовых моделей \(\mathcal{M} = \{m_1,\ldots,m_4\}\):

| Ключ | Алгоритм | Суть |
|------|----------|------|
| `RandomForest` | Random Forest | ансамбль решающих деревьев, bagging |
| `Lasso` | L1-регрессия | линейная модель с L1-штрафом |
| `Ridge` | L2-регрессия | линейная модель с L2-штрафом |
| `GradientBoosting` | градиентный бустинг деревьев | последовательное добавление слабых learners |

### 5.1 Lasso (L1)

\[
\min_{\boldsymbol{\beta}} \;
\frac{1}{2N}\sum_{i=1}^{N}(y_{i,t} - \mathbf{z}_i^\top \boldsymbol{\beta})^2
+ \alpha \|\boldsymbol{\beta}\|_1
\]

Параметр \(\alpha = 0.1\) (фиксирован в коде).

### 5.2 Ridge (L2)

\[
\min_{\boldsymbol{\beta}} \;
\frac{1}{2N}\sum_{i=1}^{N}(y_{i,t} - \mathbf{z}_i^\top \boldsymbol{\beta})^2
+ \alpha \|\boldsymbol{\beta}\|_2^2
\]

Параметр \(\alpha = 1.0\).

### 5.3 Random Forest

Усреднение предсказаний \(B\) деревьев:

\[
\hat{y}^{\mathrm{RF}}_{i,t} = \frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{z}_i),
\quad B = 500
\]

Каждое дерево обучается на bootstrap-выборке и случайном подмножестве признаков (`max_features='sqrt'`).

### 5.4 Gradient Boosting

Аддитивная модель:

\[
\hat{y}^{\mathrm{GB}}_{i,t} = \sum_{b=1}^{B} \eta \, h_b(\mathbf{z}_i),
\quad B = 500,\; \eta = 0.05
\]

где \(h_b\) — очередное регрессионное дерево, подбираемое по остаткам.

---

## 6. Optimized Voting Ensemble

Для таргета \(t\) итоговое предсказание — **взвешенная сумма** базовых моделей:

\[
\hat{y}_{i,t} = \sum_{k=1}^{4} w_{t,k}\, \hat{y}^{(k)}_{i,t},
\qquad
w_{t,k} \ge 0,\;
\sum_{k=1}^{4} w_{t,k} = 1
\]

где \(\hat{y}^{(k)}_{i,t} = m_k(\mathbf{z}_i)\) — предсказание \(k\)-й базовой модели.

В sklearn это `VotingRegressor(estimators=[...], weights=[w_{t,1},\ldots,w_{t,4}])`.

Сборка сохраняется в `models/adaptive_ensemble.pkl` → `model_data['models'][target_name]`.

---

## 7. Оптимизация весов \(w_{t,k}\)

На этапе обучения для каждого \(t\):

1. Train разбивается на **train_main** (80%) и **val_opt** (20%), `random_state=42`.
2. Каждая базовая модель \(m_k\) обучается на train_main и даёт вектор предсказаний на val_opt: \(\hat{\mathbf{y}}^{(k)}_{\mathrm{val}}\).
3. Ищутся веса, минимизирующие **MAE** на val_opt:

\[
\mathbf{w}_t^\* = \arg\min_{\mathbf{w} \ge 0}
\mathrm{MAE}\!\left(
\mathbf{y}_{\mathrm{val},t},\;
\sum_{k=1}^{4} w_k \hat{\mathbf{y}}^{(k)}_{\mathrm{val}}
\right)
\]

с нормировкой \(\sum_k w_k = 1\) (через \(w_k \leftarrow |w_k| / \sum |w_k|\)).

Метод: **L-BFGS-B** (`scipy.optimize.minimize`), границы \(w_k \in [0,1]\).

После нахождения \(\mathbf{w}_t^\*\) финальный `VotingRegressor` дообучается на **полном** train_main+val_opt (внутренний split только для подбора весов).

**Следствие:** для разных осей \(\mathbf{w}_t^\*\) различаются. Оптимизатор может «схлопнуть» ансамбль в одну модель (вес 1.0 у одного \(m_k\), у остальных 0), если это минимизирует MAE на val.

---

## 8. Цикл обучения

```text
data/processed/train.csv  ──┐
data/processed/validation.csv ┘
        │
        ▼
prepare_training_data_split()
  • φ_eng, φ_cross (без fit)
  • imputer.fit(train); scaler.fit(train)
  • transform(val)
        │
        ▼
train_and_evaluate_adaptive_ensembles()
  для каждого t ∈ {left_x, left_y, left_z, right_x, right_y, right_z}:
    • optimize_ensemble_weights → w_t*
    • VotingRegressor.fit → f_t
        │
        ▼
save_model() → adaptive_ensemble.pkl
```

---

## 9. Инференс

Для нового пациента с сырыми признаками \(\mathbf{x}\):

\[
\hat{\boldsymbol{\delta}}_i =
\begin{pmatrix}
f_{\mathrm{left\_x}}(\mathbf{z}_i) \\
f_{\mathrm{left\_y}}(\mathbf{z}_i) \\
f_{\mathrm{left\_z}}(\mathbf{z}_i) \\
f_{\mathrm{right\_x}}(\mathbf{z}_i) \\
f_{\mathrm{right\_y}}(\mathbf{z}_i) \\
f_{\mathrm{right\_z}}(\mathbf{z}_i)
\end{pmatrix}
\]

Предсказанная lateral-позиция центра (для визуализации):

\[
\hat{\mathbf{p}}^{\mathrm{lateral}} = \mathbf{p}^{\mathrm{supine}} + \hat{\boldsymbol{\delta}}
\]

(компонентно для каждой почки и оси).

API: `POST /predict` → `kidney_displacement_api.py` → `predict_targets()`.

---

## 10. Метрики качества (при обучении)

На holdout-части (validation CSV) для каждого таргета \(t\):

**MAE (мм):**

\[
\mathrm{MAE}_t = \frac{1}{N}\sum_{i=1}^{N} \left| y_{i,t} - \hat{y}_{i,t} \right|
\]

**RMSE (мм):**

\[
\mathrm{RMSE}_t = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (y_{i,t} - \hat{y}_{i,t})^2}
\]

**R²:**

\[
R^2_t = 1 - \frac{\sum_i (y_{i,t} - \hat{y}_{i,t})^2}{\sum_i (y_{i,t} - \bar{y}_t)^2}
\]

**Векторная ошибка почки** (для клинической оценки, не отдельный loss при обучении):

\[
e^{(L)}_i = \left\| \boldsymbol{\delta}^{(L)}_i - \hat{\boldsymbol{\delta}}^{(L)}_i \right\|_2,
\qquad
e^{(R)}_i = \left\| \boldsymbol{\delta}^{(R)}_i - \hat{\boldsymbol{\delta}}^{(R)}_i \right\|_2
\]

**Доля в пределах порога:**

\[
\text{within\_5mm} = \frac{1}{6N}\sum_{i,t} \mathbf{1}\big[|y_{i,t}-\hat{y}_{i,t}| \le 5\big]
\]

---

## 11. Почему оси предсказываются разными методами

Физически смещение — **один 3D-вектор** на почку, но в production используется **multi-target scalar regression**:

- разные оси имеют **разную дисперсию** и разную клиническую точность измерения (X обычно сложнее, ~4–5 mm MAE);
- линейные модели (Lasso/Ridge) лучше работают на «малых» компонентах Y/Z;
- нелинейные (RF/GB) — на крупных нелинейных эффектах в X;
- оптимизация \(\mathbf{w}_t^\*\) **отдельно по MAE** позволяет подобрать разный состав ансамбля для каждой скалярной координаты.

Multi-output регрессия одним векторным выходом **не используется**.

---