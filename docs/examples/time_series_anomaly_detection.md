# Time series anomaly detection

## Overview

Anomaly detection in temporal data requires careful balance between sensitivity and specificity, particularly when
implementing Temporal Deep Belief Networks (tDBNs). The effectiveness of these models hinges on precise configuration of
several critical parameters:

Structural Parameters:

- num_parents: Controls network complexity (typically 1-5)
- markov_lag: Governs temporal dependency range (usually 1-3 steps) (greatly increase computation time)
- Non_stationary: force algorithm to compute num_transition matrices. If the process is stationary, 
only one matrix is required.

Processing Parameters:

- artificial_slicing: Enables window-based temporal analysis
- artificial_slicing_params: Defines window size and stride characteristics

Our example demonstrates that optimal performance is achieved not through maximal complexity, but through strategic
parameter calibration. Special attention will be given to artificial slicing.

The dataset was taken from [here](https://www.timeseriesclassification.com/description.php?Dataset=ECG200).

## Set up

```python
import numpy as np

from applybn.anomaly_detection.dynamic_anomaly_detector.fast_time_series_detector import FastTimeSeriesDetector
from applybn.anomaly_detection.dynamic_anomaly_detector.data_formatter import TemporalDBNTransformer

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pyts.approximation import SymbolicAggregateApproximation # not installed by default
from sklearn.metrics import f1_score

sns.set_theme()

np.random.seed(51)
```

## Step 1. Load data
```python
df = pd.read_csv("data/anomaly_detection/ts/ECG200.csv")
df = df.loc[:, df.columns[1::3].tolist() + ["anomaly"]] # take each 3 step for faster computations
print(df.shape) # (200, 33)
```

## Step 2. Artificial slicing
This example demonstrates the role of appropriate hyperparameters for slicing.

```python
y = df.pop("anomaly")

fracs_normal = []
fracs_anomaly = []
x = np.linspace(1, 20, 20).astype(int)
pbar = tqdm(x)

for i in pbar:
    pbar.set_description(f"Processing {i}")
    transformer = TemporalDBNTransformer(window=i, stride=i)
    df_ = transformer.transform(df, y)

    try:
        non_anom_frac = df_["anomaly"].value_counts(normalize=True)[0]
    except KeyError:
        non_anom_frac = 0
        
    fracs_normal.append(non_anom_frac)
    fracs_anomaly.append(1 - non_anom_frac)

final_df = pd.DataFrame({"normal": fracs_normal,
                         "anomaly": fracs_anomaly},
                        index=x)

ax = sns.lineplot(data=final_df)
ax.set(title="Stride=Window size", xlabel='Window size', ylabel='Fraction')
ax.set_xticks(x)
plt.show()
```
![img.png](sliding_variations.png)

So it is very important to keep balance in target vector. 

## Step 3. Data processing
```python
transformer = TemporalDBNTransformer(window=5, stride=1)
df_ = transformer.transform(df, y)

y = df_.pop("anomaly")
```

## Step 4. SAX
```python
transformer = SymbolicAggregateApproximation()
sax_vals = transformer.transform(df_.iloc[:, 1:])
df_ = df_.astype(str)

df_.iloc[:, 1:] = sax_vals
```

## Step 5. Detection

```python
detector = FastTimeSeriesDetector(markov_lag=1, num_parents=1)

detector.fit(df_)
detector.calibrate(y)
preds_cont = detector.predict(df_)

print(f1_score(y, preds_cont)) # 0.9171270718232044
```

## Variations

```python
for i in range(2, 6, 2):
    df = pd.read_csv("data/anomaly_detection/ts/ECG200.csv")
    df = df.loc[:, df.columns[1::3].tolist() + ["anomaly"]]

    y = df.pop("anomaly")

    transformer = TemporalDBNTransformer(window=i, stride=1)
    df_ = transformer.transform(df, y)

    y = df_.pop("anomaly")

    transformer = SymbolicAggregateApproximation()
    sax_vals = transformer.transform(df_.iloc[:, 1:])
    df_ = df_.astype(str)

    df_.iloc[:, 1:] = sax_vals

    detector = FastTimeSeriesDetector(markov_lag=1,
                                      num_parents=1)
    detector.fit(df_)
    detector.calibrate(y, verbose=0)
    preds_cont = detector.predict(df_)
    print(i)
    print(f1_score(y, preds_cont))
    print("____")
```

Result:
```
Evaluating network with LL score.
2
0.6907894736842105
____
Evaluating network with LL score.
4
0.8774928774928775
____
Evaluating network with LL score.
6
0.9487870619946092
____
```