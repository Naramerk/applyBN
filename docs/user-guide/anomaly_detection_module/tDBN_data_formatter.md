# Форматировщик данных tDBN

## Обзор
Для использования [`FastTimeSeriesDetector`](../../api/anomaly_detection/ts_anomaly_detection.md)
может потребоваться искусственная предварительная обработка данных путем их нарезки на окна. Кратко этот процесс можно описать
с помощью этого рисунка:

![img.png](sliding_window_and_stride.png)

*Техника скользящего окна. Источник: [статья](https://www.researchgate.net/figure/Sliding-window-technique_fig2_346510102)*


!!! note

    Длина скользящего окна - это параметр `window`, а `stride` - это шаг скользящего окна.

## Стратегии агрегации

После нарезки не совсем ясно, как агрегировать аномалии по субъектам.
Реализована только стратегия "any", то есть аномальным считается субъект, у которого есть хотя бы 1 аномальный шаг внутри.

## Использование
[`TemporalDBNTransformer`](../../api/anomaly_detection/tDBN_data_formatter.md) был разработан для выполнения такой операции:

```python
import numpy as np
from applybn.anomaly_detection.dynamic_anomaly_detector.data_formatter import TemporalDBNTransformer
import pandas as pd

from tabulate import tabulate # не устанавливается по умолчанию

np.random.seed(51)

def print_df(df):
    print(tabulate(df, tablefmt="github", headers="keys", showindex="always"))

df = pd.DataFrame(
    {"col1": np.linspace(0, 5, 10),
     "col2": np.linspace(5, 10, 10),
     "anomaly": np.random.choice([0, 1], p = [0.7, 0.3], size=10),}
).astype(int)

print_df(df)
print("\n\n")
label = df.pop("anomaly")
transformer = TemporalDBNTransformer(window=5, stride=2, include_label=True)

print_df(transformer.transform(df, label))
```

Что превращает:

|    |   col1 |   col2 |   anomaly |
|----|--------|--------|-----------|
|  0 |      0 |      5 |         0 |
|  1 |      0 |      5 |         0 |
|  2 |      1 |      6 |         0 |
|  3 |      1 |      6 |         0 |
|  4 |      2 |      7 |         0 |
|  5 |      2 |      7 |         1 |
|  6 |      3 |      8 |         0 |
|  7 |      3 |      8 |         0 |
|  8 |      4 |      9 |         0 |
|  9 |      5 |     10 |         0 |

В:

|    |   subject_id |   col1__0 |   col2__0 |   col1__1 |   col2__1 |   col1__2 |   col2__2 |   col1__3 |   col2__3 |   col1__4 |   col2__4 |   anomaly |
|----|--------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
|  0 |            0 |         0 |         5 |         0 |         5 |         1 |         6 |         1 |         6 |         2 |         7 |         0 |
|  1 |            1 |         1 |         6 |         1 |         6 |         2 |         7 |         2 |         7 |         3 |         8 |         1 |
|  2 |            2 |         2 |         7 |         2 |         7 |         3 |         8 |         3 |         8 |         4 |         9 |         1 |