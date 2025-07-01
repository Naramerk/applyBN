# Каузальный анализ на основе вмешательств

[Ссылка на руководство пользователя](../../user-guide/explainable_module/interventional_analysis.md)

::: applybn.explainable.causal_analysis.InterventionCausalExplainer


# Пример

```python
from applybn.explainable.causal_analysis import InterventionCausalExplainer
from sklearn.ensemble import RandomForestClassifier

interpreter = InterventionCausalExplainer()
interpreter.interpret(RandomForestClassifier(), X_train, y_train, X_test, y_test)
```

