# tDBN Data Formatter 

To use [`FastTimeSeriesDetector`]() you may need to artificially preprocess data by cutting them into windows.

You can do this with [`TemporalDBNTransformer`]():

```python
import pandas as pd
import numpy as np

from applybn.anomaly_detection.dynamic_anomaly_detector.data_formatter import TemporalDBNTransformer

df = pd.DataFrame(
    {"col1": np.linspace(0, 5, 6),
     "col2": np.linspace(5, 10, 6),
     "anomaly": np.random.choice([0, 1], p = [0.8, 0.2], size=6),}
).astype(int)

label = df.pop("anomaly")
transformer = TemporalDBNTransformer(window=5, include_label=True)

print(transformer.transform(df, label))
```

Which turns: 

|    |   col1 |   col2 |   anomaly |
|----|--------|--------|-----------|
|  0 |      0 |      5 |         0 |
|  1 |      1 |      6 |         1 |
|  2 |      2 |      7 |         0 |
|  3 |      3 |      8 |         0 |
|  4 |      4 |      9 |         0 |
|  5 |      5 |     10 |         0 |

Into:

|    |   subject_id |   col1__0 |   col2__0 |   anomaly__0 |   col1__1 |   col2__1 |   anomaly__1 |
|----|--------------|-----------|-----------|--------------|-----------|-----------|--------------|
|  0 |            0 |         0 |         5 |            0 |         1 |         6 |            1 |
|  1 |            1 |         1 |         6 |            1 |         2 |         7 |            0 |
|  2 |            2 |         2 |         7 |            0 |         3 |         8 |            0 |
|  3 |            3 |         3 |         8 |            0 |         4 |         9 |            0 |
|  4 |            4 |         4 |         9 |            0 |         5 |        10 |            0 |