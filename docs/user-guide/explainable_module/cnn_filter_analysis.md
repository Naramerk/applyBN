# CausalCNNExplainer: Mathematical Background

## Overview
The [`CausalCNNExplainer`](../../api/explainable/cnn_filter_importance.md) implements a causal inference approach to understanding convolutional neural networks
as described in the paper
[Explaining Deep Learning Models using Causal Inference by T. Narendra et al.](https://arxiv.org/pdf/1811.04376).
This module treats CNN filters as variables in a causal graph, allowing for the measurement of
filter importance through structural equation modeling rather than traditional correlation-based approaches.

## Mathematical Foundation

### Causal Representation of CNN Filters
The module represents CNN filters as nodes in a Directed Acyclic Graph (DAG),
where edges represent causal relationships between filters in consecutive layers:

$$
G = (V, E)
$$

where $V$ is the set of filters across all layers, and $E$ represents the causal connections between them.

### Structural Equation Modeling
For each filter $i$ in layer $l$, its activation $F_i^l$ is modeled as a function of
its parent filters from the previous layer:

$$F_i^l = f_i(\{F_j^{l-1} | j \in \text{parents}(i)\}) + \epsilon_i$$

The module implements this using linear regression, where each filter's output is
predicted from its parent filters' outputs:

$$F_i^l = \beta_0 + \sum_{j \in \text{parents}(i)} \beta_j \cdot F_j^{l-1} + \epsilon_i$$

### Filter Importance Calculation
The causal importance of a filter is determined by the absolute magnitudes of its regression
coefficients when predicting all child filters in the next layer:

$$\text{Importance}(F_j^l) = \sum_{i \in \text{children}(j)} |\beta_{j \rightarrow i}|$$

where $\beta_{j \rightarrow i}$ is the regression coefficient reflecting how much
filter $j$ in layer $l$ influences filter $i$ in layer $l+1$.

### Filter Pruning Strategy
The module leverages these importance scores for pruning, removing filters with the lowest causal impact:

$$\text{PruneSet} = \{F_j^l | \text{Importance}(F_j^l) \leq \text{threshold}_l\}$$

where $\text{threshold}_l$ is determined by the desired pruning percentage.

## Applications

This causal approach to CNN interpretation offers:

1. **Interpretable pruning** - Removes filters based on causal impact rather than magnitude or variance
2. **Heatmap visualization** - Shows which image regions causally affect model decisions
3. **Filter relationships** - Provides insights into inter-filter dependencies across layers

The method demonstrates that pruning based on causal importance
typically preserves model accuracy better than random pruning,
as it identifies and maintains the filters with the strongest causal influence on the network's output.

## Example

``` py title="examples/explainable/cnn_filter_importance.py"
--8<-- "examples/explainable/cnn_filter_importance.py"
```
