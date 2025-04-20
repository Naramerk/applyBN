# InterventionCausalExplainer: Mathematical Background

## Overview
The [`InterventionCausalExplainer`](../../api/explainable/interventions_causal.md) implements
a causal inference approach to model interpretation,
establishing feature importance through direct intervention rather than traditional correlation-based methods.
This module bridges machine learning interpretability with causal inference techniques to
determine how features causally impact model confidence and uncertainty.

## Mathematical Foundation

### Causal Forest for Treatment Effect Estimation
The core mathematical concept utilizes the Double Machine Learning (DML) framework with Causal Forests.
For each feature $X_j$, we estimate its causal effect on model confidence:

$$\tau(X_j) = \mathbb{E}[C | do(X_j = x)] - \mathbb{E}[C]$$

where $C$ represents model confidence and $do(X_j = x)$ is Pearl's do-operator indicating intervention.
The DML approach employs orthogonalization to remove confounding effects:

$$\tau(X_j) = \mathbb{E}[C - \mathbb{E}[C|X_{-j}] | X_j = x] - \mathbb{E}[X_j - \mathbb{E}[X_j|X_{-j}]]$$

### Aleatoric Uncertainty Quantification
The module leverages Data-IQ to compute model confidence and aleatoric uncertainty.
Aleatoric uncertainty represents the inherent noise in the data and is estimated by:

$$U_A = \mathbb{E}[\text{Var}(Y|X)]$$

where $Y$ is the target variable and $X$ are the features.

### Intervention Analysis
The module performs direct interventions by sampling new values for high-impact
features from their original distribution range:

$$X_j^{new} \sim \text{Uniform}(\min(X_j), \max(X_j))$$

This creates a counterfactual dataset to measure before/after changes in model confidence and uncertainty,
providing a more robust understanding of feature importance than traditional methods.

## Applications
This approach is particularly valuable in high-stakes decision systems where understanding causal relationships between
features and model behavior is critical.
By identifying which features causally affect model confidence,
it enables more targeted data collection and model improvement strategies.

## Example

``` py title="examples/explainable/intervention_explainer.py"
--8<-- "examples/explainable/intervention_explainer.py"
```
