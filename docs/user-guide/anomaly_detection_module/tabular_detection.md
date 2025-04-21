# Tabular anomaly detection

## Overview

Tabular anomaly detection in `applybn` implements unsupervised outlier detection by estimating scores. 
Scores are managed by [TabularDetector](../../api/anomaly_detection/tabular_outliers.md) object.
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

### Bayesian (Belief) Networks

**Bayesian Networks** (also known as **Bayes Nets** or **Belief Networks**) are probabilistic graphical models 
that represent a collection of random variables and their conditional dependencies using a **directed acyclic graph** (DAG).
In these graphs, each **node** corresponds to a random variable, and each **edge** represents a direct probabilistic dependency.

These networks provide a compact and intuitive way to encode joint probability distributions. 
Rather than modeling the full joint distribution explicitly, which becomes intractable as the number of variables grows, 
Bayesian Networks break it down using the **chain rule of probability** and the **conditional independencies** encoded
in the graph. This makes them particularly well-suited for reasoning under uncertainty, performing inference, 
and making decisions based on incomplete or noisy data.

At the heart of a Bayesian Network are two components:

1. **Structure** – the graph itself, which defines the dependencies among variables.
2. **Parameters** – the **conditional probability distributions (CPDs)** associated with each node, given its parents.

While the structure tells us *which variables influence each other*, 
the parameters quantify *how strong those influences are*. 
Together, they fully specify the joint distribution over the variables.

---

### Learning in Bayesian Networks

When constructing Bayesian Networks from data, there are two fundamental tasks:

- **Structure Learning**: 
Discovering the optimal graph structure that captures the dependencies among variables. 
This is often a combinatorially hard problem, especially for large datasets, 
as it involves searching through the space of all possible DAGs.
  
- **Parameter Learning**: 
Estimating the CPDs for each node given the structure. 
If the structure is known and data is complete, this step is relatively straightforward and 
can be done using maximum likelihood estimation or Bayesian estimation.

Learning both structure and parameters from data allows Bayesian Networks to be applied in real-world domains
where expert-designed structures are not available. 
This makes them especially valuable in areas such as bioinformatics, medical diagnosis, and automated decision systems.

!!! note

    By default `bamt` (and `applybn`) will use K2 as structure scorer and maximum likelihood as parameters scorer.

---

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

They compute an **anomaly score** as the **negative log-likelihood**:

\[
S_{\text{BN}}(\mathbf{x}) = -\sum_{i=1}^{n} \log P(x_i \mid \text{Pa}_i(\mathbf{x}))
\]

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
  \alpha \text{IQR}_i^{(j)} = Q_3 - Q_1
  \), where alpha is hyperparameter

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

---


## Usage

### With target variable
```python
import pandas as pd
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector

# run from applybn root or change path here
data = pd.read_csv("applybn/anomaly_detection/data/tabular/ecoli.csv")

detector = TabularDetector(target_name="y")
detector.fit(data)

detector.get_info(as_df=False)

print(detector.predict(data))
```

### Without target
```python
import pandas as pd
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector

# run from applybn root or change path here
data = pd.read_csv("applybn/anomaly_detection/data/tabular/ecoli.csv")

detector = TabularDetector()
detector.fit(data)

detector.get_info(as_df=False)

# print(detector.predict(data)) # raise an error
print(detector.predict_scores(data))
```
### Plotting result
```python
import pandas as pd
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector

# run from applybn root or change path here
data = pd.read_csv("applybn/anomaly_detection/data/tabular/ecoli.csv")

detector = TabularDetector(target_name="y")
detector.fit(data)

# detector.get_info(as_df=False)

preds = detector.predict(data)
detector.plot_result(preds) # scores or labels
```

### Changing methods
```python
import pandas as pd
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector

# run from applybn root or change path here
data = pd.read_csv("applybn/anomaly_detection/data/tabular/ecoli.csv")

detector = TabularDetector(target_name="y", model_estimation_method="iqr") # works because ecoli cont (as bams log says as well)
detector.fit(data)
preds = detector.predict_scores(data)
print(preds[:5])
```

#### Wrong usage
```python
import pandas as pd
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector

# run from applybn root or change path here
data = pd.read_csv("applybn/anomaly_detection/data/tabular/ecoli.csv")

detector = TabularDetector(target_name="y", model_estimation_method="cond_ratio")
detector.fit(data)
preds = detector.predict_scores(data)  # error
```

### Original modified
```python
import pandas as pd
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector

# run from applybn root or change path here
data = pd.read_csv("applybn/anomaly_detection/data/tabular/ecoli.csv")

detector = TabularDetector(target_name="y", 
                           model_estimation_method="original_modified")
detector.fit(data)
preds = detector.predict_scores(data)
```

## Example

``` py title="examples/anomaly_detection/tabular.py"
--8<-- "examples/anomaly_detection/tabular.py"
```
