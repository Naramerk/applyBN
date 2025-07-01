# Оценки как внутренние компоненты детектора

Детекторы - это просто менеджеры над объектами оценок.
Они представляют собой способы оценки каждой строки в каждой подвыборке.

!!! warning

    Оценки имеют ограниченную поддержку типов данных.

`applybn` реализует две основные группы оценок: на основе модели и на основе близости.

Подробнее читайте в [Руководстве пользователя](../../user-guide/anomaly_detection_module/tabular_detection.md).

---

# Оценка
::: applybn.anomaly_detection.scores.score.Score

---

# Оценки на основе близости
## Оценка локальных выбросов
::: applybn.anomaly_detection.scores.proximity_based.LocalOutlierScore

## Оценка на основе Isolation Forest
::: applybn.anomaly_detection.scores.proximity_based.IsolationForestScore

---

# Оценки на основе модели
::: applybn.anomaly_detection.scores.model_based.ModelBasedScore

## Оценка на основе BN
::: applybn.anomaly_detection.scores.model_based.BNBasedScore

## Оценка на основе IQR
::: applybn.anomaly_detection.scores.model_based.IQRBasedScore

## Оценка на основе отношения условных вероятностей
::: applybn.anomaly_detection.scores.model_based.CondRatioScore

## Комбинированная оценка на основе IQR и отношения условных вероятностей
::: applybn.anomaly_detection.scores.model_based.CombinedIQRandProbRatioScore