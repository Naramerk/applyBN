# Оценщики, конвейеры и байесовские сети (продвинутый уровень)

## Обзор
Оценщики предоставляют универсальную обертку для `bamt` в формате, совместимом с `sklearn`, что позволяет разработчикам беспрепятственно работать с байесовскими сетями. Цель состоит в реализации гибких интерфейсов оценщиков.

Фреймворк `applybn` следует этой структуре:
```
Байесовские сети (Ядро: bamt)
         |
         ↓
  Оценщики BN (Обертка для сетей bamt) [Уровень разработчика]
         |
         ↓
     Конвейеры (Интерфейс applybn для bamt) [Уровень пользователя]
```

## Байесовские сети
В `bamt` существует три типа сетей:

- `HybridBN`
- `ContinuousBN`
- `DiscreteBN`

Кроме того, `bamt` может автоматически определять типы данных.

### Поддерживаемые типы данных:
```python
# Дискретные типы
categorical_types = ["str", "O", "b", "categorical", "object", "bool"]
# Дискретные числовые типы
integer_types = ["int32", "int64"]
# Непрерывные типы
float_types = ["float32", "float64"]
```
!!! danger

    **Пожалуйста, не используйте этот уровень, если в этом нет необходимости.**


## Оценщики
Оценщики предназначены для разработчиков и наследуются от `sklearn.BaseEstimator`.

### Определение типа BN
Используйте этот статический метод для определения `bn_type` на основе пользовательских данных:
```python
estimator = BNEstimator()
data = pd.read_csv(DATA_PATH)
estimator.detect_bn(data)
```

### Метод Fit
```python
estimator = BNEstimator()
data = pd.read_csv(DATA_PATH)
descriptor = {"types": {...}, "signs": {...}}
preprocessed_data = preprocessor([...]).apply(data)

fit_package = (preprocessed_data, descriptor, data)
estimator.fit(fit_package)
```

#### Обучение структуры и параметров (`partial=False`, по умолчанию)
##### Низкоуровневый подход:
```python
bn = HybridBN(use_mixture=False, has_logit=True)  # Может быть любой тип сети
bn.add_nodes(descriptor)
bn.add_edges(X, scoring_function=("MI", ))
bn.fit_parameters(clean_data)
```
##### Использование `applybn`:
```python
estimator = BNEstimator(use_mixture=False, has_logit=True,
                        learning_params={"scoring_function": ("MI", )})
<...>
estimator.fit(fit_package)
```

#### Обучение только структуры (`partial="structure"`)
##### Низкоуровневый подход:
```python
bn.add_nodes(descriptor)
bn.add_edges(X)
```
##### Использование `applybn`:
```python
estimator = BNEstimator(partial="structure")
<...>
estimator.fit(fit_package)
```

#### Обучение только параметров (`partial="parameters"`)

!!! note

    Этот метод требует предварительно обученной структуры. Если `estimator.bn_` не установлен или `estimator.bn_.edges` пуст, будет вызвано исключение `NotFittedError`.

##### Низкоуровневый подход:
```python
bn.fit_parameters(clean_data)
```
##### Использование `applybn`:
```python
estimator = BNEstimator(partial="parameters")
<...>
estimator.fit(fit_package)
```

### Настройка оценщиков
Вы можете создавать собственные оценщики с помощью метапрограммирования и наследования:
```python
from applybn.core.estimators.estimator_factory import EstimatorPipelineFactory
import inspect

factory = EstimatorPipelineFactory(task_type="classification")
estimator_with_default_interface = factory.estimator.__class__

class StaticEstimator(estimator_with_default_interface):
    def __init__(self):
        pass

my_estimator = StaticEstimator()
print(*inspect.getmembers(my_estimator), sep="\n")  # Проверка доступных методов
```

### Стратегия делегирования
Если метод неизвестен `BNEstimator`, он будет делегирован `self.bn_`. Если `bn_` не установлен, будет вызвано исключение `NotFittedError` или `AttributeError`.

## Конвейеры
Конвейеры объединяют `bamt_preprocessor` и `BNEstimator`. Вы можете заменить препроцессор любым трансформером, совместимым с `scikit-learn`.

Конвейеры следуют паттерну фабрики, что означает, что любой вызов `getattr` делегируется последнему шагу конвейера (обычно `BNEstimator`).

### Инициализация фабрики
Укажите тип задачи (классификация или регрессия) при инициализации:
```python
interfaces = {"classification": ClassifierMixin,
              "regression": RegressorMixin}
```
```python
import pandas as pd
from applybn.core.estimators.estimator_factory import EstimatorPipelineFactory

X = pd.read_csv(DATA_PATH)
# y = X.pop("anomaly")  # Извлечение целевой переменной, если необходимо

factory = EstimatorPipelineFactory(task_type="classification")
```

### Создание конвейера
```python
# Конвейер с препроцессором по умолчанию
pipeline = factory()
# Конвейер с пользовательским препроцессором
# pipeline = factory(preprocessor)

pipeline.fit(X)
```

### Управление атрибутами конвейера
Чтобы сначала обучить структуру, выполнить промежуточные шаги, а затем обучить параметры:
```python
# Обучение структуры с использованием функции оценки MI
pipeline = factory(partial="structure", learning_params={"scoring_function": "MI"})
pipeline.fit(X)

# Промежуточные шаги обработки
<...>

# Обучение параметров
pipeline.set_params(bn_estimator__partial="parameters")
pipeline.fit(X)

print(pipeline.bn_.edges)
```

#### Установка параметров
Компоненты фабрики конвейера структурированы следующим образом:
```python
CorePipeline([("preprocessor", wrapped_preprocessor),
              ("bn_estimator", estimator)])
```
Чтобы изменить атрибуты препроцессора:
```python
pipeline.set_params(preprocessor__attrName=value)
```
!!! tip

    Параметры могут быть установлены во время инициализации или после создания конвейера.

### Стратегия делегирования
Методы конвейера делегируют вызовы последнему шагу (`BNEstimator`).
```python
factory = EstimatorPipelineFactory(task_type="classification")
pipeline = factory()
pipeline.fit(X)

pipeline.get_info(as_df=False)
pipeline.save("mybn")
```

## Создание пользовательских препроцессоров
Чтобы создать пользовательский препроцессор, убедитесь, что он совместим с `scikit-learn`, унаследовав от `BaseEstimator` и `TransformerMixin`.
```python
from sklearn.base import BaseEstimator, TransformerMixin

class PreprocessorWrapper(BaseEstimator, TransformerMixin):
       <...>
       def transform(self, X):
            df = do_smth(X)
            return df
```

