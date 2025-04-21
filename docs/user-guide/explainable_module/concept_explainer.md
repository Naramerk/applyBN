# ConceptCausalExplainer: Mathematical Background

## Overview
The [`ConceptCausalExplainer`](../../api/explainable/concepts_causal.md) implements a novel approach to model
interpretation through concept-level causal analysis,
as outlined in the paper
["Concept-Level Model Interpretation From the Causal Aspect"](https://ieeexplore.ieee.org/abstract/document/9904301).
Rather than analyzing individual feature importance, this module identifies high-level concepts
within data and estimates their causal effects on model behavior,
offering interpretability that aligns with human-understandable concepts.

## Mathematical Foundation

### Concept Identification and Extraction
The module uses a two-stage approach to concept discovery:

1. **Clustering-based Concept Discovery**: The data space is partitioned using KMeans clustering:
   $$C_i = \{x_j \in D \mid \arg\min_k \|x_j - \mu_k\|^2 = i\}$$
   where $C_i$ represents cluster $i$, $D$ is the discovery dataset, and $\mu_k$ are cluster centroids.

2. **Discriminative Concept Validation**: Each cluster is validated by training a linear SVM:
   $$S_i(x) = \text{sign}(w_i^T x + b_i)$$
   A cluster is considered a valid concept if it can be discriminated from the natural dataset with AUC > threshold.

### Concept Space Representation
The feature space is transformed into a concept space:
$$A(x) = [A_1(x), A_2(x), ..., A_m(x)]$$
where $A_i(x) = 1$ if $x$ belongs to concept $i$, and 0 otherwise.

### Causal Effect Estimation
For a binary outcome $L_f$, the causal effect of concept $A_i$ is estimated using:
$$\tau_i = \mathbb{E}[L_f \mid do(A_i = 1)] - \mathbb{E}[L_f \mid do(A_i = 0)]$$

For continuous outcomes like model confidence, the Double Machine Learning framework is used:
$$\tau_i(x) = \mathbb{E}[Y \mid do(A_i = 1), X = x] - \mathbb{E}[Y \mid do(A_i = 0), X = x]$$

where $Y$ is the outcome of interest (e.g., model confidence) and $X$ are other concepts acting as controls.

## Applications

This approach offers several advantages:
1. **Interpretability**: Concepts correspond to human-understandable patterns in data
2. **Causal Understanding**: Estimates of causal effects rather than correlations
3. **Diagnostic Power**: Identifies which concepts causally impact model behavior
4. **Transferability**: Concepts can be applied across different models on the same data

By understanding causal relationships between concepts and model behavior,
users can make more informed decisions about model improvements, data collection, and feature engineering strategies.

## Example

``` py title="examples/explainable/concept_explainer.py"
--8<-- "examples/explainable/concept_explainer.py"
```
