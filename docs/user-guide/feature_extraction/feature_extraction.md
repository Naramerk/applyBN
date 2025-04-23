# BNFeatureGenerator: Mathematical Background

## Overview

The [`BNFeatureGenerator`](../../api/feature_extraction/feature_extraction.md) implements a novel approach to feature engineering through Bayesian Networks, as described in the paper ["Bayesian feature construction for the improvement of classification performance"](https://www.researchgate.net/publication/339293950_Bayesian_feature_construction_for_the_improvement_of_classification_performance) by Manolis Maragoudakis. This module automatically constructs probabilistic lambda-features derived from Bayesian Network inference, enabling complex relationship modeling that can enhance model performance.

## Mathematical Foundation

### Bayesian Network Construction

The algorithm consists of several key steps:

1. **Structure Learning**: A directed acyclic graph (DAG) is learned to represent probabilistic relationships between variables

2. **Parameter Learning**: The module fits conditional probability distributions for each node

### Feature Generation through Probabilistic Inference

The transformation process produces lambda-features by:

1. For each original feature $X_i$, generating a corresponding lambda-feature $\lambda_i$
2. For discrete features: $\lambda_i = P(X_i = x_i | Pa(X_i) = pa_i)$
3. For continuous features: $\lambda_i = P(X_i \leq x_i | Pa(X_i) = pa_i)$

These lambda-features capture complex probabilistic relationships that linear models cannot express, encoding the conditional likelihood of observed values given their parents.

## Example

The following example demonstrates how to use BNFeatureGenerator on the banknote authentication dataset:

``` py title="examples/feature_extraction/banknote-authentication_example.py"
--8<-- "examples/feature_extraction/banknote_authentication_example.py"
```

The BNFeatureGenerator enhances model performance by capturing complex probabilistic relationships in the data, improving the classifier's ability to distinguish between classes, as demonstrated in the example above. 
