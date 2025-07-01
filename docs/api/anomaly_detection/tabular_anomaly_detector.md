# Обнаружение выбросов в табличных данных

Детектор для обнаружения аномалий в данных без учителя.
Этот модуль состоит из детекторов, которые управляют объектами [scores](scores.md),
поскольку в этом методе каждая строка имеет свою собственную оценку аномальности.

Подробнее читайте в [Руководстве пользователя](../../user-guide/anomaly_detection_module/tabular_detection.md).

---

# Детектор
::: applybn.anomaly_detection.static_anomaly_detector.tabular_detector.TabularDetector

## Пример

```python
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector

X = load_data() # любым способом
detector = TabularDetector()
detector.fit(X)

detector.predict_scores(X)
```
