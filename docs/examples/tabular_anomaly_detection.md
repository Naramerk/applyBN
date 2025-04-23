# Tabular Data anomaly detection

This example demonstrates how different preprocessing techniques affect
the performance of various outlier detection algorithms, with evaluation
using both classification reports and ROC-AUC metrics.

In general case you can follow this [guide](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html). 
But sometimes the result can be different due to model specification.

## Set up
```python
from tabulate import tabulate

def print_df(df):
    print(tabulate(df, tablefmt="github", headers="keys", showindex="always"))

import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector
```

## Step 1. Load data and define required variables
```python
data = pd.read_csv("../../applybn/anomaly_detection/data/tabular/ecoli.csv")
X = data.drop(columns=['y'])
y = data['y']

# Define preprocessing methods
preprocessors = {
    'No preprocessing': None,
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'PowerTransformer': PowerTransformer(method='yeo-johnson')
}

# Define outlier detection methods
detectors = {
    'IQR-based': TabularDetector(target_name='y', model_estimation_method='iqr'),
    'Isolation Forest': TabularDetector(target_name='y', additional_score='IF'),
    'Local Outlier Factor': TabularDetector(target_name='y', additional_score='LOF')
}
```

## Step 2. Train-test split
Don't forget to stratify to avoid samples disbalance!

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

## Step 3. Comparison
```python
results = []

for preproc_name, preprocessor in preprocessors.items():
    if preprocessor:
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
    else:
        X_train_processed = X_train.values
        X_test_processed = X_test.values

    # Create dataframes for the detector
    train_df = pd.DataFrame(X_train_processed, columns=X.columns)
    train_df['y'] = y_train.values
    test_df = pd.DataFrame(X_test_processed, columns=X.columns)
    test_df['y'] = y_test.values

    for det_name, detector in detectors.items():
        print(f"\n=== {preproc_name} + {det_name} ===")

        # Train and predict
        detector.fit(train_df)
        try:
            preds = detector.predict(test_df)
        except ValueError:
            detector.y_ = test_df['y'].values
            preds = detector.predict(test_df)

        scores = detector.decision_function(test_df) if hasattr(detector, 'decision_function') else None

        # Get classification report
        report = classification_report(y_test, preds, output_dict=True)
        f1 = report['1']['f1-score']

        # Calculate ROC-AUC if scores are available

        roc_auc = roc_auc_score(y_test, scores)

        # Store results
        results.append({
            'Preprocessor': preproc_name,
            'Detector': det_name,
            'F1': f1,
            'ROC-AUC': roc_auc
        })
```


## Display results table
```python
results_df = pd.DataFrame(results)

print_df(results_df.pivot(index='Detector', columns='Preprocessor', values=['F1', 'ROC-AUC']))
```


| Detector             |   ('F1', 'No preprocessing') |   ('F1', 'PowerTransformer') |   ('F1', 'RobustScaler') | ('F1', 'StandardScaler') | ('ROC-AUC', 'No preprocessing') |   ('ROC-AUC', 'PowerTransformer') |   ('ROC-AUC', 'RobustScaler') |   ('ROC-AUC', 'StandardScaler') |
|----------------------|------------------------------|------------------------------|--------------------------|--------------------------|---------------------------------|-----------------------------------|-------------------------------|---------------------------------|
| IQR-based            |                     0.5      |                    0.0645161 |                0.0952381 | **0.8**                  | **1**                           |                          0.969388 |                      0.748299 |                        0.727891 |
| Isolation Forest     |                     0        |                    0         |                0         | 0.4                      | 0.87415                         |                          0.945578 |                      0.94898  |                        0.972789 |
| Local Outlier Factor |                     0.666667 |                    0.0631579 |                0.08      | 0.666667                 | **1**                           |                          0.418367 |                      0.64966  |                        0.989796 |
