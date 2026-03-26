
# ФИНАЛЬНЫЙ ОТЧЕТ СРАВНЕНИЯ МОДЕЛЕЙ

**Дата:** 2026-03-26 11:01:59
**Источники данных:** DICOMS + Vybor + KiTS19

## 📊 РЕЗУЛЬТАТЫ МОДЕЛЕЙ


### RandomForest

- **Train MAE:** 4.223632447349657 мм
- **Validation MAE:** 14.802 мм
- **Features:** 34
- **Targets:** 78
- **Train Samples:** 239
- **Validation Samples:** 68


### XGBoost

- **Train MAE:** 0.3234293509825798 мм
- **Validation MAE:** 15.004 мм
- **Features:** 34
- **Targets:** 78
- **Train Samples:** 239
- **Validation Samples:** 68


### Adaptive_Ensemble

- **Train MAE:** nan мм
- **Validation MAE:** 1.706 мм
- **Features:** 41
- **Targets:** 6
- **Train Samples:** 245
- **Validation Samples:** 62


## 🏆 ПОБЕДИТЕЛЬ

**Adaptive_Ensemble** с Validation MAE = 1.706 мм


## 💡 РЕКОМЕНДАЦИИ

1. **Для Production:** Используйте модель с минимальным Validation MAE
2. **Для Research:** Экспериментируйте с различными комбинациями признаков
3. **Для Monitoring:** Регулярно проверяйте качество на новых данных
4. **Для Improvement:** Соберите больше данных для обучения

## 📈 СЛЕДУЮЩИЕ ШАГИ

- Увеличить объем тренировочных данных
- Оптимизировать гиперпараметры
- Попробовать нейронные сети
- Добавить временные признаки
- Улучшить предобработку данных

---
*Отчет сгенерирован автоматически*
