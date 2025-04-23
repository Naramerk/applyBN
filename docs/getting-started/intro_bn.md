# Bayesian (Belief) Networks

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

# Learning in Bayesian Networks

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