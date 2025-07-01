# Ресемплинг на основе байесовских сетей

[Ссылка на руководство пользователя](../../user-guide/oversampling_module/bn_oversampling.md)

::: applybn.imbalanced.over_sampling.BNOverSampler

# Пример

```python
from applybn.imbalanced.over_sampling import BNOverSampler

# Инициализация с BN на основе GMM (автоматически настраивается через use_mixture=True)
oversampler = BNOverSampler(
    class_column='target', 
    strategy='max_class'  # Соответствие размеру самого большого класса
)

# Генерация выборок с использованием P(X|class) из обученной BN
X_res, y_res = oversampler.fit_resample(X, y)
```