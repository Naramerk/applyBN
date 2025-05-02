# Tabular anomaly detection

## Overview

Tabular anomaly detection in `applybn` implements unsupervised outlier detection by estimating scores. 
Scores are managed by [TabularDetector](../../api/anomaly_detection/tabular_anomaly_detector.md) object.
The basis of this algorithm was taken 
from ["Effective Outlier Detection based on Bayesian Network and Proximity"](https://ieeexplore.ieee.org/document/8622230). 

`applybn` contains a several novel modification of this the algorithm such as different scores, 
mitigation of normality limitations, dealing with mixed data.

The list of changes:

- Added support for mixed data
- Added several new methods such as IQR, Cond Ratio Probability that based on arbitrary bayesian network.

Please consider this table when choosing appropriate algorithm:

| Data type  | Allowed methods                                   |
|------------|---------------------------------------------------|
| Continuous | `iqr`, `original_modified`                        |
| Discrete   | `cond_ratio`, `original_modified`                 |
| Mixed      | `cond_ratio + iqr (default)`, `original_modified` |

## Mathematical Foundation

### Original Method 

The original method uses **Bayesian Networks with proximity-based enhancement** for outlier detection.

##### Bayesian Network Modeling

Given a dataset \( \mathcal{D} = \{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)}\} \), 
where each row \( \mathbf{x}^{(j)} = (x_1^{(j)}, x_2^{(j)}, \dots, x_n^{(j)}) \), 
the authors first **learn the Bayesian Network** structure and parameters:

- Structure: DAG \( G \) where each node \( X_i \) has parent set \( \text{Pa}(X_i) \)
- Parameters: CPDs for each \( X_i \mid \text{Pa}(X_i) \), usually Gaussian if variables are continuous.

##### Likelihood-Based Row Evaluation

Once the network is trained, the likelihood of each row \( \mathbf{x} \) is evaluated as:

\[
P(\mathbf{x}) = \prod_{i=1}^{n} P(x_i \mid \text{Pa}_i(\mathbf{x}))
\]

Where \( \text{Pa}_i(\mathbf{x}) \) are the values of \( \text{Pa}(X_i) \) in the same row.

##### Scoring Rows

They compute an **anomaly score** as follows.

If the CPDs are Gaussian:

\[
P(x_i \mid \text{Pa}_i(\mathbf{x})) = \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left( -\frac{(x_i - \mu_i(\text{Pa}_i))^2}{2\sigma_i^2} \right)
\]

So the score becomes:

\[
S_{\text{BN}}(\mathbf{x}) = \sum_{i=1}^{n} \frac{(x_i - \mu_i(\text{Pa}_i(\mathbf{x})))^2}{2\sigma_i^2}
\]

---

##### Proximity-Based Enhancement

To improve the detection (especially for local anomalies), they add a **proximity-based score**.
They define a **combined anomaly score** like:

\[
S_{\text{Total}}(\mathbf{x}) = S_{\text{BN}}(\mathbf{x}) + S_{\text{prox}}(\mathbf{x})
\]

Where \( S_{prox}(x) \) is negative factors from LOF.

---

### Original modified method

Given a set of variables \( X = \{X_1, X_2, \dots, X_n\} \), a Bayesian Network defines the joint distribution as:

\[
P(X) = \prod_{i=1}^{n} P(X_i \mid \text{Pa}(X_i))
\]

Where:

- \( \text{Pa}(X_i) \) denotes the parents of node \( X_i \) in the DAG.
- Each \( P(X_i \mid \text{Pa}(X_i)) \) is a **Conditional Probability Distribution (CPD)**.

Let:

- \( \mathcal{D} = \{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)}\} \)
- \( \text{pa}_i^{(j)} = \text{pa}_i(\mathbf{x}^{(j)}) \), the parent values of \( X_i \) in row \( j \)

Then the **local score** for node \( X_i \) in row \( j \) is:

\[
s_i^{(j)} = \text{Score}(x_i^{(j)} \mid \text{pa}_i^{(j)})
\]

This score is computed by comparing \( x_i^{(j)} \) to **other values of \( X_i \)** that share similar parent values \( \text{pa}_i^{(j)} \). You can express this using a conditional group:

\[
s_i^{(j)} = \phi \left( x_i^{(j)}, \{ x_i^{(k)} \mid \text{pa}_i^{(k)} = \text{pa}_i^{(j)} \} \right)
\]

Where \( \phi \) is a comparing scoring function.

---

### IQR (Interquartile range) method

For a given row \( j \):

- \( y = x_i^{(j)} \)
- Conditional set:
  \(
  \mathcal{C}_i^{(j)} = \{ x_i^{(k)} \mid \text{pa}_i^{(k)} = \text{pa}_i^{(j)} \}
  \)

- IQR bounds for node \( i \) under parent configuration \( \text{pa}_i^{(j)} \):
  \(
  Q_1 = \text{Quantile}_{25\%}(\mathcal{C}_i^{(j)}), \quad Q_3 = \text{Quantile}_{75\%}(\mathcal{C}_i^{(j)})
  \)

- IQR:
  \(
  \text{IQR}_i^{(j)} = \alpha \cdot (Q_3 - Q_1)
  \), where $\alpha$ is hyperparameter

- Bounds:
  \(
  \text{Lower}_i^{(j)} = Q_1, \quad \text{Upper}_i^{(j)} = Q_3
  \)

- Reference distances:
  \(
  d_{\min} = \min(\mathcal{C}_i^{(j)}), \quad d_{\max} = \max(\mathcal{C}_i^{(j)})
  \)

---

### IQR-Based Score Function

Now define the node score \( s_i^{(j)} \) as:

\[
s_i^{(j)} =
\begin{cases}
0, & \text{if } \text{Lower}_i^{(j)} < x_i^{(j)} \leq \text{Upper}_i^{(j)} \\[6pt]
\min\left(1, \dfrac{|x_i^{(j)} - c|}{|d|} \right), & \text{otherwise}
\end{cases}
\]

Where:

- \( c = \arg\min_{b \in \{\text{Lower}_i^{(j)}, \text{Upper}_i^{(j)}\}} |x_i^{(j)} - b| \)
- \( d = \begin{cases}
d_{\max}, & \text{if } c = \text{Upper}_i^{(j)} \\
d_{\min}, & \text{if } c = \text{Lower}_i^{(j)} \\
\end{cases} \)

---

### Conditional Ratio Probability
Scores based on how **unexpected** a value is under its conditional distribution compared to the marginal distribution.

Let:

- \( X_i \) be the node, and \( x_i^{(j)} \) its value in row \( j \)
- \( \text{pa}_i^{(j)} \) be the parent configuration for that row
- \( \mathcal{D} \) be the dataset

We define:

- **Marginal probability** of value \( x_i^{(j)} \):

\[
P(x_i^{(j)}) = \frac{\#(x_i = x_i^{(j)})}{N}
\]

- **Conditional probability** of value given parents:

\[
P(x_i^{(j)} \mid \text{pa}_i^{(j)}) = \frac{\#(x_i = x_i^{(j)} \land \text{pa}_i = \text{pa}_i^{(j)})}{\#(\text{pa}_i = \text{pa}_i^{(j)})}
\]

Now the local score for node \( X_i \) and row \( j \) is:

\[
s_i^{(j)} =
\begin{cases}
\text{NaN}, & \text{if } \frac{P(x_i^{(j)})}{P(x_i^{(j)} \mid \text{pa}_i^{(j)})} \notin \mathbb{R} \\
\min\left(1, \frac{P(x_i^{(j)})}{P(x_i^{(j)} \mid \text{pa}_i^{(j)})} \right), & \text{otherwise}
\end{cases}
\]

This score measures how **surprising** a value is given its context in the network — higher values imply **less expected** behavior conditionally.

## Example

``` py title="examples/anomaly_detection/tabular.py"
--8<-- "examples/anomaly_detection/tabular.py"
```
