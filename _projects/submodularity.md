---
layout: page
title: Submodularity in Multimodal Fusion
category: ongoing
categories: math
importance: 1
math: true
header-includes:
  - \usepackage{mathrsfs}
  - \usepackage{amsbsy}
bibliography:
- ref.bib
---

Method
======

Problem setup
-------------

Similar to [@perez2019mfas], we assume access to pretrained unimodal
models for each modality with respect to supervised outcome target set
$\mathbf{Y}$. Without loss of generality, suppose we have two modalities
sets denoted as $\mathbf{X}$ and $\mathbf{Z}$, and their corresponding
unimodal models $f(\cdot)$ on $X \in \mathbf{X}$ and $g(\cdot)$ on
$Z \in \mathbf{Z}$ with $M$ and $N$ layers respectively. We denote
$\mathbf{X}_i$ ($0 \leq i \leq M$) and $\mathbf{Z}_j$
($0 \leq j \leq N$) as the set of the activation of $i$th and $j$th
layer of the two corresponding modalities
($\mathbf{X}_0 \coloneqq \mathbf{X}$,
$\mathbf{Z}_0 \coloneqq \mathbf{Z}$). In the Experiments section, we
will show how the method works for more than two modalities.

With pretrained unimodal models' representation space
$\{\mathbf{X}_0, \cdots, \mathbf{X}_M \} \times \{\mathbf{Z}_0, \cdots, \mathbf{Z}_N\}$,
traditional late fusion approaches often work on fusion mechanisms of
the last representations $X_M \in \mathbf{X}_M$ and
$Z_N \in \mathbf{Z}_N$, while early fusion models generally focus on
representation learning from the primal inputs $\mathbf{X}_0$ and
$\mathbf{Z}_0$. To generalize the process of selecting representations
from unimodal pipelines, we define a fusion function
$\Phi_{ij} \coloneqq \phi(X_i, Z_j)$ as the fused representation of
$X_i$ and $Z_j$,
$\mathbf{\Phi} \subseteq \{\Phi_{00}, \Phi_{01}, \cdots, \Phi_{MN}\}$ and a *multimodal representation
selection function* $s(\cdot)$:

::: {.definition}
**Definition 1** (multimodal representation selection function).
$$\begin{aligned}
\label{eq:representation_selection_func}
    s(\mathbf{\Phi}, \mathcal{u}) \coloneqq \mathop{\mathrm{arg\,max}}_{\mathbf{\Phi}^\prime \subseteq \mathbf{\Phi}}
    u(\mathbf{\Phi}^\prime)\end{aligned}$$
:::

where $u$ is the *utility function*. A valid utility function
$u \coloneqq 2^{\mathbf{\Phi}} \xrightarrow{} \mathbb{R}$ evaluates the expressivity of the input. We use the notation $2^\mathbf{\Phi}$
for the power set of $\mathbf{\Phi}$. When
$\mathbf{\Phi} =\{\Phi_{00} \}$, eq
[\[eq:representation_selection_func\]](#eq:representation_selection_func){reference-type="ref"
reference="eq:representation_selection_func"} becomes early fusion; when
$\mathbf{\Phi} =\{\Phi_{MN} \}$, eq
[\[eq:representation_selection_func\]](#eq:representation_selection_func){reference-type="ref"
reference="eq:representation_selection_func"} is late fusion. In both
cases, $\mathbf{\Phi^\prime}$ has single unique feasible solution, making the
optimization problem in eq
[\[eq:representation_selection_func\]](#eq:representation_selection_func){reference-type="ref"
reference="eq:representation_selection_func"} trivial. We are interested
in the most general case where
$\mathbf{\Phi} \coloneqq \{\Phi_{00}, \Phi_{01}, \cdots, \Phi_{MN}\}$

Greedy Maximization on $\epsilon$-Approximately Submodular Utility Function
---------------------------------------------------------------------------

Searching through all possible
$\mathbf{\Phi}^\prime \subseteq \mathbf{\Phi}$ to find the optimal
solution of
[\[eq:representation_selection_func\]](#eq:representation_selection_func){reference-type="ref"
reference="eq:representation_selection_func"} is NP-hard due to the
combinatorial nature of the feasible set. To reduce the computational
complexity of
[\[eq:representation_selection_func\]](#eq:representation_selection_func){reference-type="ref"
reference="eq:representation_selection_func"}, we add additional
assumption on the submodularity of the utility function:

::: {.definition}
**Definition 2** (Submodularity [@NemhauserSubmodular]). For any
arbitrary set $\mathbf{S}$, a function
$f\coloneqq 2^{\mathbf{S}} \xrightarrow{} \mathbb{R}$ is submodular if
$\forall \mathbf{A} \subseteq \mathbf{B} \subseteq \mathbf{S}$ and
$e \in \mathbf{S} \backslash \mathbf{B}$,
$f(\mathbf{A} \cup \{e\}) - f(\mathbf{A}) \ge f(\mathbf{B} \cup \{e\}) - f(\mathbf{B})$
:::

With such a constraint on the submodularity of the utility function,
[@NemhauserSubmodular] proposed a Greedy Maximization algorithm
[\[algo:greedymax\]](#algo:greedymax){reference-type="ref"
reference="algo:greedymax"} to approximate the optimal solution with
pseudo-polynomial time complexity. [@NemhauserSubmodular] proved that
algorithm [\[algo:greedymax\]](#algo:greedymax){reference-type="ref"
reference="algo:greedymax"} has the following approximation guarantee:

::: {.theorem}
**Theorem 1** (Approximation guarantee of algorithm
[\[algo:greedymax\]](#algo:greedymax){reference-type="ref"
reference="algo:greedymax"} [@NemhauserSubmodular]). Suppose algorithm
[\[algo:greedymax\]](#algo:greedymax){reference-type="ref"
reference="algo:greedymax"} runs $p$ iterations to select
$\widehat{\mathbf{S}_p}$ from $2^\mathbf{S}$, then for a submodualr
function $f$, we have $$\begin{aligned}
    f(\widehat{\mathbf{S}_p}) \ge (1-e^{-\frac{p}{q}})\max_{\mathbf{S}^\prime \subseteq \mathbf{S}, |\mathbf{S}^\prime| \le q} f(\mathbf{S}^\prime)\end{aligned}$$
:::

[@ChengSubmodular] further studied the cases where the utiliy function
is not strictly submodular, but follows an $\epsilon$-approximately
submodular property:

::: {.definition}
**Definition 3** ($\epsilon$-Approximate Submodularity
[@ChengSubmodular]). For any arbitrary set $\mathbf{S}$, a function
$f\coloneqq 2^{\mathbf{S}} \xrightarrow{} \mathbb{R}$ is
$\epsilon$-approximately submodular if $\exists  \epsilon$ such that
$\forall \mathbf{A} \subseteq \mathbf{B} \subseteq \mathbf{S}$ and
$e \in \mathbf{S} \backslash \mathbf{B}$,
$f(\mathbf{A} \cup \{e\}) - f(\mathbf{A}) + \epsilon \ge f(\mathbf{B} \cup \{e\}) - f(\mathbf{B})$
:::

[@ChengSubmodular] proved the following approximation guarantee on
applying algorithm
[\[algo:greedymax\]](#algo:greedymax){reference-type="ref"
reference="algo:greedymax"} to an an $\epsilon$-approximately submodular
utility function:

::: {.theorem}
**Theorem 2** (Approximation guarantee of algorithm
[\[algo:greedymax\]](#algo:greedymax){reference-type="ref"
reference="algo:greedymax"} on $\epsilon$-approximately submodular
utility functions [@ChengSubmodular]). Suppose algorithm
[\[algo:greedymax\]](#algo:greedymax){reference-type="ref"
reference="algo:greedymax"} runs $p$ iterations to select
$\widehat{\mathbf{S}_p}$ from $2^\mathbf{S}$, then for an
$\epsilon$-approximately submodular function $f$, we have
$$\begin{aligned}
    f(\widehat{\mathbf{S}_p}) \ge (1-e^{-\frac{p}{q}})\max_{\mathbf{S}^\prime \subseteq \mathbf{S}, |\mathbf{S}^\prime| \le q} f(\mathbf{S}^\prime) - q\epsilon\end{aligned}$$
:::

::: {#co:approxsub .corollary}
**Corollary 1**. Suppose algorithm
[\[algo:greedymax\]](#algo:greedymax){reference-type="ref"
reference="algo:greedymax"} runs $p$ iterations to select
$\widehat{\mathbf{S}_p}$ from $2^\mathbf{S}$, then for an
$\epsilon$-approximately submodular function $f$, we have
$$\begin{aligned}
    f(\widehat{\mathbf{S}_p}) \ge (1-e^{-\frac{p}{|\mathbf{S}|}})\max_{\mathbf{S}^\prime \subseteq \mathbf{S}} f(\mathbf{S}^\prime) - |\mathbf{S}|\epsilon\end{aligned}$$
:::

$\widehat{\mathbf{S}_0} = \emptyset$

$X = \mathop{\mathrm{arg\,max}}_{X\in \mathbf{S} \backslash \widehat{\mathbf{S}_{i-1}} }f( \widehat{\mathbf{S}_{i-1}} \cup \{X\}) - f( \widehat{\mathbf{S}_{i-1}})$
$\widehat{\mathbf{S}_{i}} =  \widehat{\mathbf{S}_{i-1}} \cup \{X\}$

FuseBox
-------

Based on the existing theoretic guarantees on the greedy selection of
components to achieve suboptimal solution to the maximization of the
utiliy function, we propose a multimodal fusion method, FuseBox, which
applies algorithm
[\[algo:greedymax\]](#algo:greedymax){reference-type="ref"
reference="algo:greedymax"} on $\mathbf{\Phi}$ with mutual information
$\text{I}(\cdot; \mathbf{Y}) \coloneqq 2^\mathbf{\Phi} \xrightarrow{} \mathbb{R}$
as the utility function, $\mathbf{Y}$ being the label set. When we
denote the input $\mathbf{X}$ and $\mathbf{Y}$ of the mutual information
$\text{I}(\mathbf{X}; \mathbf{Y})$ as a set, we assume that the sets
follow a data distribution $\text{Dist}_{\mathbf{X}}$ and
$\text{Dist}_\mathbf{Y}$ respectively, and
$\text{I}(\mathbf{X}; \mathbf{Y}) \coloneqq \text{I}(X; Y)|_{X \sim \text{Dist}_{\mathbf{X}}, Y \sim \text{Dist}_\mathbf{Y}}$

::: {.definition}
**Definition 4** (FuseBox). Given $k$ modality random variables
$X^{(i)}|_{i \in \{1, \cdots, k\}}$, each $X^{(i)}$ has a pretrained
unimodal function $f^{(i)}(\cdot)$ that produce $M^{(i)}$ intermediate
states, let
$\mathbf{\Phi} \coloneqq \prod_{i \in \{1, \cdots, k\}}\{X^{(i)}_m|_{m \in \{0, \cdots, M^{(i)}\}}\}$.
All the $X^{(i)}$ share the same label space $\mathbf{Y}$. FuseBox gives
a subset $\widehat{\mathbf{\Phi}}_p \subseteq \mathbf{\Phi}$ in
pseudo-polynomial time complexity that approximately achieves
$\max_{\mathbf{\Phi}^\prime \in \mathbf{\Phi}} \text{I}(\mathbf{\Phi}^\prime; \mathbf{Y})$
by running algorithm
[\[algo:greedymax\]](#algo:greedymax){reference-type="ref"
reference="algo:greedymax"} on $\mathbf{\Phi}$ in $p$ iterations with
$\text{I}(\cdot; \mathbf{Y})$ as the utility function.
:::

The submodularity of mutual information is proved in [@ChengSubmodular]:

::: {.theorem}
**Theorem 3**
(\[\$\\epsilon\$-Approximate Submodularity of mutual information \\cite{ChengSubmodular}\]).
$\exists  \epsilon$ such that
$\forall \mathbf{A} \subseteq \mathbf{B} \subseteq \mathbf{\Phi}$ and
$e \in \mathbf{\Phi} \backslash \mathbf{B}$,
$\text{I}(\mathbf{A} \cup \{e\};\mathbf{Y}) - \text{I}(\mathbf{A}; \mathbf{Y}) + \epsilon \ge \text{I}(\mathbf{B} \cup \{e\};\mathbf{Y}) - \text{I}(\mathbf{B};\mathbf{Y})$
:::

Combining the $\epsilon$-approximate submodularity nature of
$\text{I}(\cdot; \mathbf{Y})$ and corollary
[Corollary 1](#co:approxsub){reference-type="ref"
reference="co:approxsub"}, the approximation performance of FuseBox is
stated as:

::: {.prop}
**Proposition 1** (Approximation Guarantee of FuseBox). Suppose
$\mathbf{\Phi}^* = s(\mathbf{\Phi}, \text{I}(\cdot; \mathbf{Y}))$,
$\widehat{\mathbf{\Phi}}_p$ is selected by algorithm
[\[algo:greedymax\]](#algo:greedymax){reference-type="ref"
reference="algo:greedymax"}, then
$\text{I}(\widehat{\mathbf{\Phi}}_p; \mathbf{Y}) \ge (1 - e^{-\frac{p}{|\mathbf{\Phi}|}}) \text{I}(\mathbf{\Phi}^*; \mathbf{Y}) - |\mathbf{\Phi}|\epsilon$
:::
