# NMIFeatureSelector: отбор признаков с помощью нормализованной взаимной информации

[Ссылка на руководство пользователя](../../user-guide/feature_selection/nmi_feature_selection.md)

::: applybn.feature_selection.bn_nmi_feature_selector.NMIFeatureSelector


# Пример

```python
from applybn.feature_selection.bn_nmi_feature_selector import NMIFeatureSelector

selector = NMIFeatureSelector(threshold=0.5, n_bins=20)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
```