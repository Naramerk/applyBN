# Scores as Detector Internals

Detectors are just managers over scores objects. 
They are a ways to evaluate each row in each subsample.

!!! warning

    Scores have limited data types support.

`applybn` implements two major group of scores: model based and proximity based.

Read more in [User Guide](../../user-guide/anomaly_detection_module/tabular_detection.md). 

---

# Score
::: applybn.anomaly_detection.scores.score.Score

---

# Proximity based scores
## Local Outlier Score
::: applybn.anomaly_detection.scores.proximity_based.LocalOutlierScore

## Isolation Forest Score
::: applybn.anomaly_detection.scores.proximity_based.IsolationForestScore

---

# Model based scores
::: applybn.anomaly_detection.scores.model_based.ModelBasedScore

## BN based Score
::: applybn.anomaly_detection.scores.model_based.BNBasedScore

## IQR Based Score
::: applybn.anomaly_detection.scores.model_based.IQRBasedScore

## Conditional Probability Ratio Score
::: applybn.anomaly_detection.scores.model_based.CondRatioScore

## Combined IQR and Conditional Probability Ratio Score
::: applybn.anomaly_detection.scores.model_based.CombinedIQRandProbRatioScore