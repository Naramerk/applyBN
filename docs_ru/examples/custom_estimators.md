# Estimators, Pipelines, and Bayesian Networks (Advanced)

## Overview
Estimators provide a universal wrapper for `bamt` with an `sklearn`-compatible format, allowing developers to work seamlessly with Bayesian networks. The goal is to implement flexible estimator interfaces.

The `applybn` framework follows this structure:
```
Bayesian Networks (Core: bamt)
         |
         ↓
  BN Estimators (Wrapper for bamt's networks) [Developer Level]
         |
         ↓
     Pipelines (applybn Interface for bamt) [User Level]
```

## Bayesian Networks
In `bamt`, there are three types of networks:

- `HybridBN`
- `ContinuousBN`
- `DiscreteBN`

Additionally, `bamt` can automatically infer data types.

### Supported Data Types:
```python
# Discrete types
categorical_types = ["str", "O", "b", "categorical", "object", "bool"]
# Discrete numerical types
integer_types = ["int32", "int64"]
# Continuous types
float_types = ["float32", "float64"]
```
!!! danger

    **Please do not use this level unless necessary.**


## Estimators
Estimators are intended for developers and inherit from `sklearn.BaseEstimator`.

### Detecting BN Type
Use this static method to determine `bn_type` based on user data:
```python
estimator = BNEstimator()
data = pd.read_csv(DATA_PATH)
estimator.detect_bn(data)
```

### Fit Method
```python
estimator = BNEstimator()
data = pd.read_csv(DATA_PATH)
descriptor = {"types": {...}, "signs": {...}}
preprocessed_data = preprocessor([...]).apply(data)

fit_package = (preprocessed_data, descriptor, data)
estimator.fit(fit_package)
```

#### Learning Structure and Parameters (`partial=False`, default)
##### Low-Level Approach:
```python
bn = HybridBN(use_mixture=False, has_logit=True)  # Can be any network type
bn.add_nodes(descriptor)
bn.add_edges(X, scoring_function=("MI", ))
bn.fit_parameters(clean_data)
```
##### Using `applybn`:
```python
estimator = BNEstimator(use_mixture=False, has_logit=True,
                        learning_params={"scoring_function": ("MI", )})
<...>
estimator.fit(fit_package)
```

#### Learning Only Structure (`partial="structure"`)
##### Low-Level Approach:
```python
bn.add_nodes(descriptor)
bn.add_edges(X)
```
##### Using `applybn`:
```python
estimator = BNEstimator(partial="structure")
<...>
estimator.fit(fit_package)
```

#### Learning Only Parameters (`partial="parameters"`)

!!! note

    This method requires a pre-trained structure. If `estimator.bn_` is not set or `estimator.bn_.edges` is empty, a `NotFittedError` is raised.

##### Low-Level Approach:
```python
bn.fit_parameters(clean_data)
```
##### Using `applybn`:
```python
estimator = BNEstimator(partial="parameters")
<...>
estimator.fit(fit_package)
```

### Customizing Estimators
You can create custom estimators using metaprogramming and inheritance:
```python
from applybn.core.estimators.estimator_factory import EstimatorPipelineFactory
import inspect

factory = EstimatorPipelineFactory(task_type="classification")
estimator_with_default_interface = factory.estimator.__class__

class StaticEstimator(estimator_with_default_interface):
    def __init__(self):
        pass

my_estimator = StaticEstimator()
print(*inspect.getmembers(my_estimator), sep="\n")  # Verify available methods
```

### Delegation Strategy
If a method is unknown to `BNEstimator`, it will be delegated to `self.bn_`. If `bn_` is not set, a `NotFittedError` or `AttributeError` is raised.

## Pipelines
Pipelines combine `bamt_preprocessor` and `BNEstimator`. You can replace the preprocessor with any `scikit-learn`-compatible transformer.

Pipelines follow a factory pattern, meaning any `getattr` call is delegated to the final pipeline step (usually `BNEstimator`).

### Initializing the Factory
Specify the task type (classification or regression) when initializing:
```python
interfaces = {"classification": ClassifierMixin,
              "regression": RegressorMixin}
```
```python
import pandas as pd
from applybn.core.estimators.estimator_factory import EstimatorPipelineFactory

X = pd.read_csv(DATA_PATH)
# y = X.pop("anomaly")  # Extract target variable if necessary

factory = EstimatorPipelineFactory(task_type="classification")
```

### Creating a Pipeline
```python
# Default preprocessor pipeline
pipeline = factory()
# Custom preprocessor pipeline
# pipeline = factory(preprocessor)

pipeline.fit(X)
```

### Managing Pipeline Attributes
To learn the structure first, perform intermediate steps, and then learn parameters:
```python
# Learn structure using MI scoring function
pipeline = factory(partial="structure", learning_params={"scoring_function": "MI"})
pipeline.fit(X)

# Intermediate processing steps
<...>

# Learn parameters
pipeline.set_params(bn_estimator__partial="parameters")
pipeline.fit(X)

print(pipeline.bn_.edges)
```

#### Setting Parameters
Pipeline factory components are structured as follows:
```python
CorePipeline([("preprocessor", wrapped_preprocessor),
              ("bn_estimator", estimator)])
```
To modify preprocessor attributes:
```python
pipeline.set_params(preprocessor__attrName=value)
```
!!! tip

    Parameters can be set during initialization or after creating the pipeline.

### Delegation Strategy
Pipeline methods delegate calls to the final step (`BNEstimator`).
```python
factory = EstimatorPipelineFactory(task_type="classification")
pipeline = factory()
pipeline.fit(X)

pipeline.get_info(as_df=False)
pipeline.save("mybn")
```

## Creating Custom Preprocessors
To create a custom preprocessor, ensure it is `scikit-learn` compatible by inheriting from `BaseEstimator` and `TransformerMixin`.
```python
from sklearn.base import BaseEstimator, TransformerMixin

class PreprocessorWrapper(BaseEstimator, TransformerMixin):
       <...>
       def transform(self, X):
            df = do_smth(X)
            return df
```

