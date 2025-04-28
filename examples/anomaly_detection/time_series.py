from applybn.anomaly_detection.dynamic_anomaly_detector.fast_time_series_detector import FastTimeSeriesDetector
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import json

def add_anomalies(df, anomaly_fraction=0.05, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    df_anomaly = df.copy()
    total = df.shape[0]

    # Initialize label matrix with zeros
    anomaly_labels = np.zeros(total, dtype=int)

    # Exclude subject_id column from anomalies
    cols_to_modify = [col for col in df.columns if col != 'subject_id']

    n_anomalies = int(df.shape[0] * anomaly_fraction)

    # Generate random positions
    rows = np.random.randint(0, df.shape[0], size=n_anomalies)

    for row_idx in rows:
        new_value = np.random.choice(["a", "b", "c"], size=1)[0]
        anomaly_labels[row_idx] = 1
        random_col_name = np.random.choice(cols_to_modify, size=1)[0]
        df_anomaly.at[row_idx, random_col_name] = new_value

    return df_anomaly, anomaly_labels

df_discrete = pd.read_csv("../../data/anomaly_detection/ts/meteor_discrete_example_data.csv")

df_anomaly, anomalies = add_anomalies(df_discrete, anomaly_fraction=0.3, random_state=42)

result = {}
for i in np.linspace(-10, 0, 100):
    for j in np.linspace(0.2, 1, 100):
        detector = FastTimeSeriesDetector(abs_threshold=i,
                                          rel_threshold=j)

        preds_cont = detector.fit_predict(df_anomaly)
        # plt.hist(detector.scores_.flatten(), bins=100)
        # plt.show()
        # raise Exception
        result[str([str(i), str(j)])] = f1_score(anomalies, preds_cont)

with open("result.json", "w+") as f:
    json.dump(result, f, indent=4)