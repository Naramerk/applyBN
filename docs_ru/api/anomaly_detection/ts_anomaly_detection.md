# Обнаружение выбросов во временных рядах

[FastTimeSeriesDetector](#detector) для обнаружения аномалий во временных рядах без учителя
на основе реализации на Java. Для интеграции этого модуля была внесена необходимая модификация.

Подробнее читайте в [Руководстве пользователя](../../user-guide/anomaly_detection_module/time_series_detection.md).

---

# Детектор
::: applybn.anomaly_detection.dynamic_anomaly_detector.fast_time_series_detector.FastTimeSeriesDetector

## Пример

```python
from applybn.anomaly_detection.dynamic_anomaly_detector.fast_time_series_detector import FastTimeSeriesDetector

X = load_data() # любым способом
y_true = X.pop("y") # опционально
detector = FastTimeSeriesDetector()
detector.fit(X)

detector.calibrate(y_true) # опционально
detector.predict(X)
```