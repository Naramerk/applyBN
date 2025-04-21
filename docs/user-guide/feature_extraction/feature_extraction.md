# BNFeatureGenerator: Mathematical Background

## Overview

The `BNFeatureGenerator` implements a novel approach to feature engineering through Bayesian Networks, as described in the paper ["Bayesian feature construction for the improvement of classification performance"](https://www.researchgate.net/publication/339293950_Bayesian_feature_construction_for_the_improvement_of_classification_performance) by Manolis Maragoudakis. This module automatically constructs probabilistic lambda-features derived from Bayesian Network inference, enabling complex relationship modeling that can enhance model performance.

## Mathematical Foundation

### Bayesian Network Structure Learning

The algorithm consists of several key steps:

1. **Data Pre-processing**: The input data is prepared by:
   - Encoding categorical columns using a LabelEncoder
   - Discretizing continuous columns using KBinsDiscretizer with k-means strategy
   - Optionally incorporating a target variable as a node in the network

2. **Structure Learning**: A directed acyclic graph (DAG) is learned to represent probabilistic relationships between variables:
   - Nodes represent features in the dataset
   - Edges represent conditional dependencies
   - When a target variable is provided, a blacklist prevents edges from the target to predictors

3. **Parameter Learning**: The module fits conditional probability distributions for each node:
   - Discrete nodes: Categorical distributions as $P(X|Pa(X))$
   - Continuous nodes: Gaussian distributions conditioned on parent values

### Feature Generation through Probabilistic Inference

The transformation process produces lambda-features by:

1. For each original feature $X_i$, generating a corresponding lambda-feature $\lambda_i$
2. For discrete features: $\lambda_i = P(X_i = x_i | Pa(X_i) = pa_i)$
3. For continuous features: $\lambda_i = P(X_i \leq x_i | Pa(X_i) = pa_i)$

These lambda-features capture complex probabilistic relationships that linear models cannot express, encoding:
- The conditional likelihood of observed values given their parents
- The probabilistic "unexpectedness" of values in context
- Local causal neighborhood effects

## Example

The following example demonstrates how to use BNFeatureGenerator to enhance a Decision Tree classifier on the banknote authentication dataset:

```python
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from fin import BNFeatureGenerator
import ssl

# Set SSL context to avoid certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context

# Load the banknote-authentication dataset
data = fetch_openml(name='banknote-authentication', version=1, as_frame=True)
X = pd.DataFrame(data.data)
y = pd.Series(data.target, name='target')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Reset indices to ensure proper alignment
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Using only original features
print("\n1. Training Decision Tree with original features...")
dt_original = DecisionTreeClassifier(random_state=42)
dt_original.fit(X_train, y_train)
y_pred_original = dt_original.predict(X_test)

# Generating Bayesian Network features
print("\n2. Generating Bayesian Network features...")
bn_feature_generator = BNFeatureGenerator()
bn_feature_generator.fit(X=X_train, y=y_train)  # Fit the generator with training data

# Transform both training and testing data
X_train_bn = bn_feature_generator.transform(X_train).reset_index(drop=True)
X_test_bn = bn_feature_generator.transform(X_test).reset_index(drop=True)

# Combine original and Bayesian features
X_train_combined = pd.concat([X_train, X_train_bn], axis=1)
X_test_combined = pd.concat([X_test, X_test_bn], axis=1)

# Train with combined features
print("\nTraining Decision Tree with combined features...")
dt_combined = DecisionTreeClassifier(random_state=42)
dt_combined.fit(X_train_combined, y_train)
y_pred_combined = dt_combined.predict(X_test_combined)

# Compare results
print("\nClassification Report with Original Features:")
print(classification_report(y_test, y_pred_original))
print("\nClassification Report with Combined Features:")
print(classification_report(y_test, y_pred_combined))
```

The BNFeatureGenerator enhances model performance by capturing complex probabilistic relationships in the data, improving the classifier's ability to distinguish between classes, as demonstrated in the example above. 
