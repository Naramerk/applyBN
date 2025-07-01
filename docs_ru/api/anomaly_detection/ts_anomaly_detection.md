# Time Series Outlier Detection

[FastTimeSeriesDetector](#detector) for detection of anomalies in time series in unsupervised way
based on Java implementation. A required modification was concluded to integrate this module.

Read more in [User Guide](../../user-guide/anomaly_detection_module/time_series_detection.md).

---

# Detector
::: applybn.anomaly_detection.dynamic_anomaly_detector.fast_time_series_detector.FastTimeSeriesDetector

## Example 

```python
from applybn.anomaly_detection.dynamic_anomaly_detector.fast_time_series_detector import FastTimeSeriesDetector

X = load_data() # any way
y_true = X.pop("y") # optional
detector = FastTimeSeriesDetector()
detector.fit(X)

detector.calibrate(y_true) # optional
detector.predict(X)
```