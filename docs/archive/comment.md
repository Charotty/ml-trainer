### 📊 Признаки (features), которые использовать в модели

#### 🧱 Геометрия почки (относительные и нормализованные)

* kidney_left_center_x_rel

* kidney_left_center_y_rel

* kidney_left_center_z_rel

* kidney_left_center_x_norm

* kidney_left_center_y_norm

* kidney_left_center_z_norm

* kidney_right_center_x_rel

* kidney_right_center_y_rel

* kidney_right_center_z_rel

* kidney_right_center_x_norm

* kidney_right_center_y_norm

* kidney_right_center_z_norm

---

#### 📏 Размеры почек

* kidney_left_length_mm

* kidney_left_volume_cm3

* kidney_right_length_mm

* kidney_right_volume_cm3

---

#### 🧍 Геометрия тела

* body_width_mm
* body_depth_mm
* body_area_mm2

---

#### 🎯 Относительные расстояния

* kidney_left_to_spine_distance

* kidney_right_to_spine_distance

* kidney_left_to_body_center_distance

* kidney_right_to_body_center_distance

---

#### 🧭 Центры (базовые ориентиры)

* spine_center_x

* spine_center_y

* spine_center_z

* body_com_x

* body_com_y

* body_com_z

---

#### 🔄 (опционально) Положение пациента

* patient_position (закодированное: supine/lateral)

---

# 🎯 Target (что предсказывает модель)

* kidney_left_delta_x

* kidney_left_delta_y

* kidney_left_delta_z

* kidney_right_delta_x

* kidney_right_delta_y

* kidney_right_delta_z

---

# 📈 Метрики для оценки модели

#### 📊 Основные

* MAE (Mean Absolute Error) по каждой оси (мм)
* RMSE (Root Mean Squared Error) по каждой оси (мм)

---

#### 🎯 По почкам отдельно

* MAE_left_kidney
* MAE_right_kidney

---

#### 📏 Геометрическая ошибка

* Euclidean distance error (мм):
  √(Δx² + Δy² + Δz²)

---

#### 🏥 Клинические метрики

* % предсказаний с ошибкой < 5 мм
* % предсказаний с ошибкой < 10 мм

---

#### 📉 Стабильность

* Std ошибки (разброс предсказаний)

---

#### 🔍 Дополнительно

* R² score
* Median Absolute Error

---

#### ⚠️ Контроль качества

* Max error (максимальная ошибка)
* Количество выбросов (>20 мм)
