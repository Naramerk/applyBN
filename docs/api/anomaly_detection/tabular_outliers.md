# Tabular Outlier Detection

Detector for detection of anomalies in data in unsupervised way.
This module consists of detectors as a managers of [scores](scores.md) object
because each row has its own anomaly score in this method. 

Read more in [User Guide](../../user-guide/anomaly_detection_module/tabular_detection.md).

---

# Detector
::: applybn.anomaly_detection.static_anomaly_detector.tabular_detector.TabularDetector

## Example 

```python
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector

X = load_data() # any way
detector = TabularDetector()
detector.fit(X)

detector.predict_scores(X)
```
